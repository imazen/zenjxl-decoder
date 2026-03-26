// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(clippy::identity_op)]

#[cfg(feature = "avx")]
pub(super) mod avx;
#[cfg(feature = "avx512")]
pub(super) mod avx512;
#[cfg(feature = "sse42")]
pub(super) mod sse42;

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
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            #[$crate::__arcane]
            $(#[$($attr)*])*
            fn [<$dname __v4>](_token: $crate::X64V4Token, $($arg: $ty),*) $(-> $ret)? {
                $name($crate::Avx512Descriptor::from_token(_token), $($arg),*)
            }

            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            #[$crate::__arcane]
            $(#[$($attr)*])*
            fn [<$dname __v3>](_token: $crate::X64V3Token, $($arg: $ty),*) $(-> $ret)? {
                $name($crate::AvxDescriptor::from_token(_token), $($arg),*)
            }

            #[cfg(all(target_arch = "x86_64", feature = "sse42"))]
            #[$crate::__arcane]
            $(#[$($attr)*])*
            fn [<$dname __v2>](_token: $crate::X64V2Token, $($arg: $ty),*) $(-> $ret)? {
                $name($crate::Sse42Descriptor::from_token(_token), $($arg),*)
            }

            $(#[$($attr)*])*
            fn [<$dname __scalar>](_token: $crate::ScalarToken, $($arg: $ty),*) $(-> $ret)? {
                $name($crate::ScalarDescriptor::from_token(_token), $($arg),*)
            }

            $(#[$($attr)*])*
            $pub fn $dname($($arg: $ty),*) $(-> $ret)? {
                use $crate::__SimdToken;
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                if let Some(t) = $crate::X64V4Token::summon() {
                    return [<$dname __v4>](t, $($arg),*);
                }
                #[cfg(all(target_arch = "x86_64", feature = "avx"))]
                if let Some(t) = $crate::X64V3Token::summon() {
                    return [<$dname __v3>](t, $($arg),*);
                }
                #[cfg(all(target_arch = "x86_64", feature = "sse42"))]
                if let Some(t) = $crate::X64V2Token::summon() {
                    return [<$dname __v2>](t, $($arg),*);
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

        $crate::test_sse42!($name);
        $crate::test_avx!($name);
        $crate::test_avx512!($name);
    };
}

#[cfg(feature = "sse42")]
#[doc(hidden)]
#[macro_export]
macro_rules! test_sse42 {
    ($name:ident) => {
        paste::paste! {
            #[test]
            fn [<$name _sse42>]() {
                use $crate::__SimdToken;
                let Some(token) = $crate::X64V2Token::summon() else { return; };
                #[$crate::__arcane]
                fn inner(token: $crate::X64V2Token) {
                    $name($crate::Sse42Descriptor::from_token(token))
                }
                inner(token);
            }
        }
    };
}

#[cfg(feature = "avx")]
#[doc(hidden)]
#[macro_export]
macro_rules! test_avx {
    ($name:ident) => {
        paste::paste! {
            #[test]
            fn [<$name _avx>]() {
                use $crate::__SimdToken;
                let Some(token) = $crate::X64V3Token::summon() else { return; };
                #[$crate::__arcane]
                fn inner(token: $crate::X64V3Token) {
                    $name($crate::AvxDescriptor::from_token(token))
                }
                inner(token);
            }
        }
    };
}

#[cfg(feature = "avx512")]
#[doc(hidden)]
#[macro_export]
macro_rules! test_avx512 {
    ($name:ident) => {
        paste::paste! {
            #[test]
            fn [<$name _avx512>]() {
                use $crate::__SimdToken;
                let Some(token) = $crate::X64V4Token::summon() else { return; };
                #[$crate::__arcane]
                fn inner(token: $crate::X64V4Token) {
                    $name($crate::Avx512Descriptor::from_token(token))
                }
                inner(token);
            }
        }
    };
}

#[cfg(not(feature = "sse42"))]
#[doc(hidden)]
#[macro_export]
macro_rules! test_sse42 {
    ($name:ident) => {};
}

#[cfg(not(feature = "avx"))]
#[doc(hidden)]
#[macro_export]
macro_rules! test_avx {
    ($name:ident) => {};
}

#[cfg(not(feature = "avx512"))]
#[doc(hidden)]
#[macro_export]
macro_rules! test_avx512 {
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
        $crate::bench_avx512!($name, $criterion);
        $crate::bench_avx!($name, $criterion);
        $crate::bench_sse42!($name, $criterion);
        $name(
            $crate::ScalarDescriptor::new().unwrap(),
            $criterion,
            "scalar",
        );
    };
}

#[cfg(feature = "avx512")]
#[doc(hidden)]
#[macro_export]
macro_rules! bench_avx512 {
    ($name:ident, $criterion:ident) => {
        if let Some(d) = $crate::Avx512Descriptor::new() {
            d.call(|d| $name(d, $criterion, "avx512"));
        }
    };
}

#[cfg(feature = "avx")]
#[doc(hidden)]
#[macro_export]
macro_rules! bench_avx {
    ($name:ident, $criterion:ident) => {
        if let Some(d) = $crate::AvxDescriptor::new() {
            d.call(|d| $name(d, $criterion, "avx"));
        }
    };
}

#[cfg(feature = "sse42")]
#[doc(hidden)]
#[macro_export]
macro_rules! bench_sse42 {
    ($name:ident, $criterion:ident) => {
        if let Some(d) = $crate::Sse42Descriptor::new() {
            d.call(|d| $name(d, $criterion, "sse42"));
        }
    };
}

#[cfg(not(feature = "avx512"))]
#[doc(hidden)]
#[macro_export]
macro_rules! bench_avx512 {
    ($name:ident, $criterion:ident) => {};
}

#[cfg(not(feature = "avx"))]
#[doc(hidden)]
#[macro_export]
macro_rules! bench_avx {
    ($name:ident, $criterion:ident) => {};
}

#[cfg(not(feature = "sse42"))]
#[doc(hidden)]
#[macro_export]
macro_rules! bench_sse42 {
    ($name:ident, $criterion:ident) => {};
}
