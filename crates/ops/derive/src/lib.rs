//! The derive of `ops::abi::Layout`: a call structure's layout as Rust gives it.
//!
//! A native kernel package receives its call as a `#[repr(C)]` structure of the operator crate,
//! and the package's C header is generated from that definition (`ops::abi`). The header asserts
//! the structure's size and the offset of every field, so the C compiler refuses a declaration
//! that disagrees with Rust; those numbers are what this derive states. Rust has no stable way to
//! enumerate a structure's fields, so the derive reads them from the definition: the field list
//! is written once, where the structure is.

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Error, Fields, parse_macro_input};

/// Implements `crate::abi::Layout` for a `#[repr(C)]` structure with named fields: its name, and
/// each field's name with its byte offset (`core::mem::offset_of!`), in declaration order.
///
/// The implementation names `crate::abi`, so the derive serves the operator crate's own call
/// structures. A structure without `#[repr(C)]` has no layout a C declaration can share, and a
/// tuple or unit structure no field names a declaration can carry; both are refused.
#[proc_macro_derive(Layout)]
pub fn layout(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand(&input) {
        Ok(expanded) => expanded.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

fn expand(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let name = &input.ident;

    let mut c_layout = false;
    for attribute in input.attrs.iter().filter(|one| one.path().is_ident("repr")) {
        attribute.parse_nested_meta(|meta| {
            c_layout |= meta.path.is_ident("C");
            // `align(N)` and `packed(N)` carry an argument, which says nothing about whether the
            // layout is C's.
            if meta.input.peek(syn::token::Paren) {
                let argument;
                syn::parenthesized!(argument in meta.input);
                argument.parse::<proc_macro2::TokenStream>()?;
            }
            Ok(())
        })?;
    }
    if !c_layout {
        return Err(Error::new_spanned(
            name,
            "a call structure's layout is shared with C only under `#[repr(C)]`",
        ));
    }

    let Data::Struct(data) = &input.data else {
        return Err(Error::new_spanned(
            name,
            "only a structure has fields to lay out",
        ));
    };
    let Fields::Named(fields) = &data.fields else {
        return Err(Error::new_spanned(
            name,
            "a C declaration names its fields: a call structure has named fields",
        ));
    };
    let fields: Vec<_> = fields
        .named
        .iter()
        .filter_map(|field| field.ident.as_ref())
        .collect();

    Ok(quote! {
        impl crate::abi::Layout for #name {
            const NAME: &'static str = ::core::stringify!(#name);
            const FIELDS: &'static [(&'static str, usize)] = &[
                #( (::core::stringify!(#fields), ::core::mem::offset_of!(#name, #fields)) ),*
            ];
        }
    })
}
