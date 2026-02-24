// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#[cfg(feature = "wasm128")]
pub(super) mod simd128;

#[macro_export]
macro_rules! simd_function {
    (
        $dname:ident,
        $descr:ident: $descr_ty:ident,
        $(#[$($attr:meta)*])*
        $pub:vis fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty )? $body: block
    ) => {
        #[inline(always)]
        $(#[$($attr)*])*
        $pub fn $name<$descr_ty: $crate::SimdDescriptor>($descr: $descr_ty, $($arg: $ty),*) $(-> $ret)? $body

        paste::paste! {
            #[cfg(all(target_arch = "wasm32", feature = "wasm128"))]
            #[$crate::__arcane]
            $(#[$($attr)*])*
            fn [<$dname __wasm128>](_token: $crate::Wasm128Token, $($arg: $ty),*) $(-> $ret)? {
                $name($crate::Wasm128Descriptor::from_token(_token), $($arg),*)
            }

            $(#[$($attr)*])*
            fn [<$dname __scalar>](_token: $crate::ScalarToken, $($arg: $ty),*) $(-> $ret)? {
                $name($crate::ScalarDescriptor::from_token(_token), $($arg),*)
            }

            $(#[$($attr)*])*
            $pub fn $dname($($arg: $ty),*) $(-> $ret)? {
                use $crate::__SimdToken;
                #[cfg(all(target_arch = "wasm32", feature = "wasm128"))]
                if let Some(t) = $crate::Wasm128Token::summon() {
                    return [<$dname __wasm128>](t, $($arg),*);
                }
                [<$dname __scalar>]($crate::ScalarToken::summon().unwrap(), $($arg),*)
            }
        }
    };
}

#[macro_export]
macro_rules! test_all_instruction_sets {
    (
        $name:ident
    ) => {
        paste::paste! {
            #[test]
            fn [<$name _scalar>]() {
                use $crate::SimdDescriptor;
                $name($crate::ScalarDescriptor::new().unwrap())
            }
        }

        $crate::test_wasm128!($name);
    };
}

#[cfg(feature = "wasm128")]
#[doc(hidden)]
#[macro_export]
macro_rules! test_wasm128 {
    ($name:ident) => {
        paste::paste! {
            #[test]
            fn [<$name _wasm128>]() {
                use $crate::__SimdToken;
                let Some(token) = $crate::Wasm128Token::summon() else { return; };
                #[$crate::__arcane]
                fn inner(token: $crate::Wasm128Token) {
                    $name($crate::Wasm128Descriptor::from_token(token))
                }
                inner(token);
            }
        }
    };
}

#[cfg(not(feature = "wasm128"))]
#[doc(hidden)]
#[macro_export]
macro_rules! test_wasm128 {
    ($name:ident) => {};
}

#[macro_export]
macro_rules! bench_all_instruction_sets {
    (
        $name:ident,
        $criterion:ident
    ) => {
        #[allow(unused)]
        use $crate::SimdDescriptor;
        $crate::bench_wasm128!($name, $criterion);
        $name(
            $crate::ScalarDescriptor::new().unwrap(),
            $criterion,
            "scalar",
        );
    };
}

#[cfg(feature = "wasm128")]
#[doc(hidden)]
#[macro_export]
macro_rules! bench_wasm128 {
    ($name:ident, $criterion:ident) => {
        if let Some(d) = $crate::Wasm128Descriptor::new() {
            d.call(|d| $name(d, $criterion, "wasm128"));
        }
    };
}

#[cfg(not(feature = "wasm128"))]
#[doc(hidden)]
#[macro_export]
macro_rules! bench_wasm128 {
    ($name:ident, $criterion:ident) => {};
}
