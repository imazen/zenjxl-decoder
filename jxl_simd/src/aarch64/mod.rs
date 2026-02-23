// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(clippy::identity_op)]

#[cfg(feature = "neon")]
pub(super) mod neon;

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
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            #[$crate::__arcane]
            $(#[$($attr)*])*
            fn [<$dname __neon>](_token: $crate::NeonToken, $($arg: $ty),*) $(-> $ret)? {
                $name($crate::NeonDescriptor::from_token(_token), $($arg),*)
            }

            $(#[$($attr)*])*
            fn [<$dname __scalar>](_token: $crate::ScalarToken, $($arg: $ty),*) $(-> $ret)? {
                $name($crate::ScalarDescriptor::from_token(_token), $($arg),*)
            }

            $(#[$($attr)*])*
            $pub fn $dname($($arg: $ty),*) $(-> $ret)? {
                use $crate::__SimdToken;
                #[cfg(all(target_arch = "aarch64", feature = "neon"))]
                if let Some(t) = $crate::NeonToken::summon() {
                    return [<$dname __neon>](t, $($arg),*);
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

        $crate::test_neon!($name);
    };
}

#[cfg(feature = "neon")]
#[doc(hidden)]
#[macro_export]
macro_rules! test_neon {
    ($name:ident) => {
        paste::paste! {
            #[test]
            fn [<$name _neon>]() {
                use $crate::__SimdToken;
                let Some(token) = $crate::NeonToken::summon() else { return; };
                #[$crate::__arcane]
                fn inner(token: $crate::NeonToken) {
                    $name($crate::NeonDescriptor::from_token(token))
                }
                inner(token);
            }
        }
    };
}

#[cfg(not(feature = "neon"))]
#[doc(hidden)]
#[macro_export]
macro_rules! test_neon {
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
        $crate::bench_neon!($name, $criterion);
        $name(
            $crate::ScalarDescriptor::new().unwrap(),
            $criterion,
            "scalar",
        );
    };
}

#[cfg(feature = "neon")]
#[doc(hidden)]
#[macro_export]
macro_rules! bench_neon {
    ($name:ident, $criterion:ident) => {
        if let Some(d) = $crate::NeonDescriptor::new() {
            d.call(|d| $name(d, $criterion, "neon"));
        }
    };
}

#[cfg(not(feature = "neon"))]
#[doc(hidden)]
#[macro_export]
macro_rules! bench_neon {
    ($name:ident, $criterion:ident) => {};
}
