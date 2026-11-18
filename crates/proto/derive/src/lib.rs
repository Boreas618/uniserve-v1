//! The derive of `proto::reload::Reload`: how a change of each field of a configuration
//! structure takes effect, declared on the field.
//!
//! Every named field carries one `#[reload(...)]` attribute, and a field without one does not
//! compile, so no option is added without stating what its change does:
//!
//! - `#[reload(step)]`, `order`, `generation`, `program`, `epoch`, `daemon`, `storage`: a change
//!   of the field takes effect that way (`proto::reload::Effect`);
//! - `#[reload(by = path)]`: the effect depends on the old and the new value, and the function
//!   `path(previous: Option<&T>, next: &T) -> Effect` decides it (`previous` is `None` when the
//!   deployment starts from nothing);
//! - `#[reload(fields)]`: the field is a structure that derives `Reload` itself, and each of its
//!   fields declares its own effect, under the path `<field>.<inner field>`.
//!
//! The generated comparison names every field, so it never misses one, and its paths are the
//! configuration's option paths (`scheduler_config.chunked_prefill_size`).

use proc_macro::TokenStream;
use quote::quote;
use syn::spanned::Spanned;
use syn::{Data, DeriveInput, Error, Fields, Ident, parse_macro_input};

/// The effects a field may declare by name, each the variant of `proto::reload::Effect` it
/// stands for.
const EFFECTS: [(&str, &str); 7] = [
    ("step", "Step"),
    ("order", "Order"),
    ("generation", "Generation"),
    ("program", "Program"),
    ("epoch", "Epoch"),
    ("daemon", "Daemon"),
    ("storage", "Storage"),
];

/// Implements `proto::reload::Reload` for a structure with named fields from each field's
/// `#[reload(...)]` attribute (module documentation).
#[proc_macro_derive(Reload, attributes(reload))]
pub fn reload(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand(&input) {
        Ok(expanded) => expanded.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

/// How one field declares its effect.
enum Declared {
    /// A fixed effect, the `Effect` variant's name.
    Fixed(Ident),
    /// A function of the previous and the next value.
    By(syn::Path),
    /// The field's own fields declare theirs.
    Fields,
}

/// Reads the one `#[reload(...)]` attribute of `field`.
fn declared(field: &syn::Field) -> syn::Result<Declared> {
    let name = field.ident.as_ref();
    let mut found: Option<Declared> = None;
    for attribute in field
        .attrs
        .iter()
        .filter(|one| one.path().is_ident("reload"))
    {
        if found.is_some() {
            return Err(Error::new_spanned(
                attribute,
                "a field declares one effect: one `#[reload(...)]` attribute",
            ));
        }
        attribute.parse_nested_meta(|meta| {
            if meta.path.is_ident("by") {
                found = Some(Declared::By(meta.value()?.parse()?));
                return Ok(());
            }
            if meta.path.is_ident("fields") {
                found = Some(Declared::Fields);
                return Ok(());
            }
            let effect = EFFECTS
                .iter()
                .find(|(word, _)| meta.path.is_ident(word))
                .ok_or_else(|| {
                    meta.error(
                        "unknown effect: one of step, order, generation, program, epoch, daemon, \
                         storage, `by = function` or `fields`",
                    )
                })?;
            found = Some(Declared::Fixed(Ident::new(effect.1, meta.path.span())));
            Ok(())
        })?;
    }
    found.ok_or_else(|| {
        Error::new_spanned(
            name,
            "every field states how a change of it takes effect: add `#[reload(...)]`",
        )
    })
}

fn expand(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let name = &input.ident;
    let Data::Struct(data) = &input.data else {
        return Err(Error::new_spanned(
            name,
            "only a structure has fields that declare effects",
        ));
    };
    let Fields::Named(fields) = &data.fields else {
        return Err(Error::new_spanned(
            name,
            "a configuration structure names its fields, which are its option paths",
        ));
    };

    let mut compared = Vec::new();
    let mut taken = Vec::new();
    for field in &fields.named {
        let Some(ident) = field.ident.as_ref() else {
            return Err(Error::new_spanned(
                field,
                "a configuration field has a name",
            ));
        };
        let key = ident.to_string();
        let changed = quote! {
            previous.is_none_or(|previous| previous.#ident != next.#ident)
        };
        match declared(field)? {
            Declared::Fixed(effect) => {
                compared.push(quote! {
                    if #changed {
                        changes.push(::proto::reload::Change::new(
                            ::proto::reload::join(path, #key),
                            ::proto::reload::Effect::#effect,
                        ));
                    }
                });
                taken.push(quote! {
                    if effect == ::proto::reload::Effect::#effect {
                        self.#ident.clone_from(&next.#ident);
                    }
                });
            }
            Declared::By(function) => {
                compared.push(quote! {
                    if #changed {
                        let effect = #function(previous.map(|previous| &previous.#ident), &next.#ident);
                        changes.push(::proto::reload::Change::new(
                            ::proto::reload::join(path, #key),
                            effect,
                        ));
                    }
                });
                taken.push(quote! {
                    if self.#ident != next.#ident
                        && #function(Some(&self.#ident), &next.#ident) == effect
                    {
                        self.#ident.clone_from(&next.#ident);
                    }
                });
            }
            Declared::Fields => {
                compared.push(quote! {
                    ::proto::reload::Reload::reload(
                        previous.map(|previous| &previous.#ident),
                        &next.#ident,
                        &::proto::reload::join(path, #key),
                        changes,
                    );
                });
                taken.push(quote! {
                    ::proto::reload::Reload::take(&mut self.#ident, &next.#ident, effect);
                });
            }
        }
    }

    let (implemented, arguments, clauses) = input.generics.split_for_impl();
    Ok(quote! {
        impl #implemented ::proto::reload::Reload for #name #arguments #clauses {
            fn reload(
                previous: ::core::option::Option<&Self>,
                next: &Self,
                path: &str,
                changes: &mut ::std::vec::Vec<::proto::reload::Change>,
            ) {
                #(#compared)*
            }

            fn take(&mut self, next: &Self, effect: ::proto::reload::Effect) {
                #(#taken)*
            }
        }
    })
}
