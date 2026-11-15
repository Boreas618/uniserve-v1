//! Chat-template content-shape detection and formatting options.
//!
//! Hugging Face templates expect message `content` either as a plain string or
//! as an OpenAI-style list of typed parts. When the configuration states no
//! shape (`chat_config.template_content_format = "auto"`),
//! `detect_chat_template_content_format` inspects the parsed template AST:
//!
//! - no loop over a message's `content` selects `String`;
//! - such a loop selects `OpenAi`;
//! - such a loop plus an `if` condition testing `content is string` selects
//!   `Preserve`, because the template handles both shapes.
//!
//! The analysis is name-based and ignores scoping: any assignment or loop
//! anywhere in the template (including macro bodies) counts, and only plain
//! variable targets are tracked.

use std::collections::{HashSet, VecDeque};

use minijinja::machinery::ast::{Expr, ForLoop, Set, Stmt};
use minijinja::machinery::{WhitespaceConfig, parse};
use minijinja::syntax::SyntaxConfig;

pub(super) use proto::config::ChatTemplateContentFormat;

/// Returns whether an expression references the named variable.
fn is_var_access(expr: &Expr, varname: &str) -> bool {
    matches!(expr, Expr::Var(v) if v.id == varname)
}

/// Returns whether an expression is the requested string constant.
fn is_const_str(expr: &Expr, value: &str) -> bool {
    matches!(expr, Expr::Const(c) if c.value.as_str() == Some(value))
}

/// Returns whether an expression accesses the requested variable attribute.
fn is_attr_access(expr: &Expr, varname: &str, key: &str) -> bool {
    match expr {
        Expr::GetItem(g) => is_var_access(&g.expr, varname) && is_const_str(&g.subscript_expr, key),
        Expr::GetAttr(g) => is_var_access(&g.expr, varname) && g.name == key,
        _ => false,
    }
}

/// Returns whether an expression references a variable or one of its elements.
///
/// With `key`, matches `varname.key` or `varname['key']`; without it, matches
/// `varname` itself. Filters, tests, and slices applied to the match are looked
/// through, so `messages|reverse` and `messages[1:]` still count as `messages`.
fn is_var_or_elems_access(expr: &Expr, varname: &str, key: Option<&str>) -> bool {
    match expr {
        Expr::Filter(f) => f
            .expr
            .as_ref()
            .is_some_and(|inner| is_var_or_elems_access(inner, varname, key)),
        Expr::Test(t) => is_var_or_elems_access(&t.expr, varname, key),
        Expr::Slice(s) => is_var_or_elems_access(&s.expr, varname, key),
        _ => key.map_or_else(
            || is_var_access(expr, varname),
            |key| is_attr_access(expr, varname, key),
        ),
    }
}

/// The statements of a template, each before the statements its bodies hold: the one traversal
/// every analysis of this module reads.
///
/// Every statement that holds a body is descended into: loops (and their `else` bodies), both
/// branches of a condition, `with`, `set` blocks, `autoescape` and `filter` blocks, blocks, and
/// the bodies of macros and `call` blocks.
fn statements<'a>(root: &'a Stmt<'a>) -> Vec<&'a Stmt<'a>> {
    let mut statements = Vec::new();
    let mut pending = vec![root];
    while let Some(statement) = pending.pop() {
        statements.push(statement);
        let bodies: [&[Stmt<'a>]; 2] = match statement {
            Stmt::Template(template) => [&template.children, &[]],
            Stmt::ForLoop(for_loop) => [&for_loop.body, &for_loop.else_body],
            Stmt::IfCond(if_cond) => [&if_cond.true_body, &if_cond.false_body],
            Stmt::WithBlock(with_block) => [&with_block.body, &[]],
            Stmt::SetBlock(set_block) => [&set_block.body, &[]],
            Stmt::AutoEscape(auto_escape) => [&auto_escape.body, &[]],
            Stmt::FilterBlock(filter_block) => [&filter_block.body, &[]],
            Stmt::Block(block) => [&block.body, &[]],
            Stmt::Macro(macro_stmt) => [&macro_stmt.body, &[]],
            Stmt::CallBlock(call_block) => [&call_block.macro_decl.body, &[]],
            _ => [&[], &[]],
        };
        // Pushed in reverse, so that the stack yields them in the template's order.
        pending.extend(bodies.into_iter().flatten().rev());
    }
    statements
}

/// Every `set` statement of `statements`.
fn assignments<'a>(statements: &[&'a Stmt<'a>]) -> Vec<&'a Set<'a>> {
    statements
        .iter()
        .filter_map(|statement| match statement {
            Stmt::Set(set) => Some(&**set),
            _ => None,
        })
        .collect()
}

/// Every `for` loop of `statements`.
fn loops<'a>(statements: &[&'a Stmt<'a>]) -> Vec<&'a ForLoop<'a>> {
    statements
        .iter()
        .filter_map(|statement| match statement {
            Stmt::ForLoop(for_loop) => Some(&**for_loop),
            _ => None,
        })
        .collect()
}

/// Finds variables transitively assigned from a source variable or its elements.
///
/// Returns `varname` itself followed by every alias discovered breadth-first,
/// e.g. `loop_messages` for `{% set loop_messages = messages[1:] %}`.
fn aliases(assignments: &[&Set<'_>], varname: &str) -> Vec<String> {
    let mut discovered = vec![varname.to_string()];
    let mut seen = HashSet::from([varname.to_string()]);
    let mut related = VecDeque::from([varname.to_string()]);

    while let Some(related_varname) = related.pop_front() {
        for assign in assignments {
            let Expr::Var(lhs) = &assign.target else {
                continue;
            };

            if is_var_or_elems_access(&assign.expr, &related_varname, None) {
                let lhs_name = lhs.id.to_string();
                if seen.insert(lhs_name.clone()) {
                    discovered.push(lhs_name.clone());
                    if lhs_name != related_varname {
                        related.push_back(lhs_name);
                    }
                }
            }
        }
    }

    discovered
}

/// Finds loop variables that iterate over the template's message collection.
fn message_items(statements: &[&Stmt<'_>]) -> Vec<String> {
    let message_varnames = aliases(&assignments(statements), "messages");

    let mut discovered = Vec::new();
    let mut seen = HashSet::new();
    for loop_ast in loops(statements) {
        let Expr::Var(target) = &loop_ast.target else {
            continue;
        };

        if message_varnames
            .iter()
            .any(|varname| is_var_or_elems_access(&loop_ast.iter, varname, None))
        {
            let target_name = target.id.to_string();
            if seen.insert(target_name.clone()) {
                discovered.push(target_name);
            }
        }
    }

    discovered
}

/// Returns whether some loop iterates over a message item's `content`.
fn has_content_item_loop(statements: &[&Stmt<'_>], message_varnames: &[String]) -> bool {
    loops(statements).into_iter().any(|loop_ast| {
        matches!(loop_ast.target, Expr::Var(_))
            && message_varnames
                .iter()
                .any(|varname| is_var_or_elems_access(&loop_ast.iter, varname, Some("content")))
    })
}

/// Returns whether an expression tests message content for string representation.
fn expression_tests_content_string(expr: &Expr<'_>, message_varnames: &[String]) -> bool {
    match expr {
        Expr::Test(test) => {
            (test.name == "string"
                && message_varnames
                    .iter()
                    .any(|varname| is_var_or_elems_access(&test.expr, varname, Some("content"))))
                || expression_tests_content_string(&test.expr, message_varnames)
        }
        Expr::UnaryOp(unary) => expression_tests_content_string(&unary.expr, message_varnames),
        Expr::BinOp(binary) => {
            expression_tests_content_string(&binary.left, message_varnames)
                || expression_tests_content_string(&binary.right, message_varnames)
        }
        Expr::Compare(compare) => {
            expression_tests_content_string(&compare.expr, message_varnames)
                || compare
                    .ops
                    .iter()
                    .any(|op| expression_tests_content_string(&op.expr, message_varnames))
        }
        Expr::IfExpr(if_expr) => {
            expression_tests_content_string(&if_expr.test_expr, message_varnames)
                || expression_tests_content_string(&if_expr.true_expr, message_varnames)
                || if_expr
                    .false_expr
                    .as_ref()
                    .is_some_and(|expr| expression_tests_content_string(expr, message_varnames))
        }
        _ => false,
    }
}

/// Returns whether some `if` condition tests a message item's `content` with `is string`.
///
/// Only `if` statement conditions are inspected; expressions in `set` statements, output blocks
/// and loop filters are not.
fn has_content_string_test(statements: &[&Stmt<'_>], message_varnames: &[String]) -> bool {
    statements.iter().any(|statement| match statement {
        Stmt::IfCond(if_cond) => expression_tests_content_string(&if_cond.expr, message_varnames),
        _ => false,
    })
}

/// Detects the content format a Jinja chat template expects from its parsed form, read with the
/// `whitespace` handling it compiles with.
///
/// # Errors
///
/// MiniJinja's parse error when the template does not parse: the format of a template that
/// cannot compile is not guessed.
pub(super) fn detect_chat_template_content_format(
    template: &str,
    whitespace: WhitespaceConfig,
) -> Result<ChatTemplateContentFormat, minijinja::Error> {
    let ast = parse(template, "chat", SyntaxConfig {}, whitespace)?;
    let statements = statements(&ast);
    let message_varnames = message_items(&statements);

    Ok(if has_content_item_loop(&statements, &message_varnames) {
        if has_content_string_test(&statements, &message_varnames) {
            ChatTemplateContentFormat::Preserve
        } else {
            ChatTemplateContentFormat::OpenAi
        }
    } else {
        ChatTemplateContentFormat::String
    })
}
