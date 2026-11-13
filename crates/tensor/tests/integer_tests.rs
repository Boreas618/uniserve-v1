//! Arithmetic on the integer element types: the element type's own arithmetic between two
//! tensors, the operations that exist on integers alone, and a scalar as the second operand.
//!
//! The expected values are PyTorch's for the same expression: `+`, `-` and `*` wrap, the
//! quotient is `torch.div(.., rounding_mode="trunc")`, and the remainder is `torch.remainder`.

#[macro_use]
mod common;

use uniserve_tensor::{DType, Device, Result, Tensor};

/// The message of the error `result` holds.
fn refusal(result: Result<Tensor>) -> String {
    match result {
        Ok(tensor) => panic!("expected an error, got {tensor:?}"),
        Err(error) => error.to_string(),
    }
}

fn arithmetic_wraps_and_the_quotient_truncates(device: &Device) -> Result<()> {
    let lhs = Tensor::new(&[250u8, 3, 16], device)?;
    let rhs = Tensor::new(&[10u8, 5, 16], device)?;
    assert_eq!((&lhs + &rhs)?.to_vec1::<u8>()?, [4, 8, 32]);
    assert_eq!((&lhs - &rhs)?.to_vec1::<u8>()?, [240, 254, 0]);
    assert_eq!((&lhs * &rhs)?.to_vec1::<u8>()?, [196, 15, 0]);

    let lhs = Tensor::new(&[i64::MAX, i64::MIN, -7, 7, -7], device)?;
    let rhs = Tensor::new(&[1i64, -1, 2, -2, -2], device)?;
    assert_eq!(
        (&lhs + &rhs)?.to_vec1::<i64>()?,
        [i64::MIN, i64::MAX, -5, 5, -9]
    );
    // The quotient rounds toward zero; the one that overflows wraps.
    assert_eq!(
        (&lhs / &rhs)?.to_vec1::<i64>()?,
        [i64::MAX, i64::MIN, -3, -3, 3]
    );

    let lhs = Tensor::new(&[300i16, -300], device)?;
    assert_eq!((&lhs * &lhs)?.to_vec1::<i16>()?, [24464, 24464]);

    // The multiplier of a hash, as a product that keeps its low 64 bits.
    let tokens = Tensor::new(&[3i64, 1 << 40], device)?;
    let multiplier = Tensor::new(&[0x5851_F42D_4C95_7F2Di64; 2], device)?;
    assert_eq!(
        (&tokens * &multiplier)?.to_vec1::<i64>()?,
        [
            3i64.wrapping_mul(0x5851_F42D_4C95_7F2D),
            (1i64 << 40).wrapping_mul(0x5851_F42D_4C95_7F2D)
        ]
    );
    Ok(())
}

fn remainder_takes_the_sign_of_the_divisor(device: &Device) -> Result<()> {
    let dividend = Tensor::new(&[7i32, -7, 7, -7, 6, -6, 0], device)?;
    let divisor = Tensor::new(&[3i32, 3, -3, -3, 3, 3, 5], device)?;
    assert_eq!(
        dividend.remainder(&divisor)?.to_vec1::<i32>()?,
        [1, 2, -2, -1, 0, 0, 0]
    );

    // The smallest value by minus one overflows as a quotient and has no remainder.
    let dividend = Tensor::new(&[i64::MIN, i64::MIN, i64::MAX], device)?;
    let divisor = Tensor::new(&[-1i64, 7, -1], device)?;
    assert_eq!(
        dividend.remainder(&divisor)?.to_vec1::<i64>()?,
        [0, i64::MIN.rem_euclid(7), 0]
    );

    let dividend = Tensor::new(&[4294967295u32, 10, 0], device)?;
    assert_eq!(
        dividend.remainder(7u32)?.to_vec1::<u32>()?,
        [4294967295 % 7, 3, 0]
    );

    // A hash that wrapped below zero still names a row of the table.
    let mixed = Tensor::new(&[[-5i64], [12]], device)?;
    let sizes = Tensor::new(&[3i64, 7, 11], device)?;
    assert_eq!(
        mixed.broadcast_remainder(&sizes)?.to_vec2::<i64>()?,
        [[1, 2, 6], [0, 5, 1]]
    );
    Ok(())
}

fn bits_combine_and_shift(device: &Device) -> Result<()> {
    let lhs = Tensor::new(&[0b1100u8, 0xff, 0x81], device)?;
    let rhs = Tensor::new(&[0b1010u8, 0x0f, 0x01], device)?;
    assert_eq!(
        lhs.bitwise_and(&rhs)?.to_vec1::<u8>()?,
        [0b1000, 0x0f, 0x01]
    );
    assert_eq!(lhs.bitwise_or(&rhs)?.to_vec1::<u8>()?, [0b1110, 0xff, 0x81]);
    assert_eq!(
        lhs.bitwise_xor(&rhs)?.to_vec1::<u8>()?,
        [0b0110, 0xf0, 0x80]
    );

    // The bits a left shift moves past the element's width are gone.
    assert_eq!(
        lhs.bitwise_left_shift(1u8)?.to_vec1::<u8>()?,
        [0b11000, 0xfe, 0x02]
    );
    assert_eq!(
        lhs.bitwise_right_shift(&rhs.minimum(4u8)?)?
            .to_vec1::<u8>()?,
        [0, 0x0f, 0x40]
    );

    // A right shift repeats the sign bit of a signed element and clears it for an unsigned one.
    let signed = Tensor::new(&[-8i32, 8, i32::MIN, 1], device)?;
    assert_eq!(
        signed.bitwise_right_shift(1i32)?.to_vec1::<i32>()?,
        [-4, 4, i32::MIN / 2, 0]
    );
    assert_eq!(
        signed.bitwise_left_shift(31i32)?.to_vec1::<i32>()?,
        [0, 0, 0, i32::MIN]
    );
    let unsigned = Tensor::new(&[0x8000_0000u32, 0xffff_ffff], device)?;
    assert_eq!(
        unsigned.bitwise_right_shift(31u32)?.to_vec1::<u32>()?,
        [1, 1]
    );

    // One step of SplitMix64 on a state below 2^63, where the signed shift is the unsigned one.
    let state = 0x1234_5678_9abc_def0i64;
    let multiplier = 0xBF58_476D_1CE4_E5B9u64 as i64;
    let x = Tensor::new(&[state], device)?;
    let mixed =
        (x.bitwise_xor(&x.bitwise_right_shift(30i64)?)? * Tensor::new(&[multiplier], device)?)?;
    assert_eq!(
        mixed.to_vec1::<i64>()?,
        [(state ^ (state >> 30)).wrapping_mul(multiplier)]
    );

    // Each row against one mask per column.
    let rows = Tensor::new(&[[0x0fu32, 0xf0], [0xff, 0x00]], device)?;
    let masks = Tensor::new(&[0x3cu32, 0xc3], device)?;
    assert_eq!(
        rows.broadcast_bitwise_and(&masks)?.to_vec2::<u32>()?,
        [[0x0c, 0xc0], [0x3c, 0x00]]
    );
    assert_eq!(
        rows.broadcast_bitwise_or(&masks)?.to_vec2::<u32>()?,
        [[0x3f, 0xf3], [0xff, 0xc3]]
    );
    assert_eq!(
        rows.broadcast_bitwise_xor(&masks)?.to_vec2::<u32>()?,
        [[0x33, 0x33], [0xc3, 0xc3]]
    );
    assert_eq!(
        rows.broadcast_bitwise_left_shift(&Tensor::new(&[4u32, 8], device)?)?
            .to_vec2::<u32>()?,
        [[0xf0, 0xf000], [0xff0, 0]]
    );
    assert_eq!(
        rows.broadcast_bitwise_right_shift(&Tensor::new(&[4u32, 2], device)?)?
            .to_vec2::<u32>()?,
        [[0, 0x3c], [0x0f, 0]]
    );

    // A view reads the same elements at its own strides.
    let grid = Tensor::arange(0i64, 6, device)?.reshape((2, 3))?;
    assert_eq!(
        grid.t()?
            .bitwise_xor(&grid.t()?.contiguous()?)?
            .to_vec2::<i64>()?,
        [[0, 0], [0, 0], [0, 0]]
    );
    Ok(())
}

fn an_integer_operation_on_floats_is_an_error(device: &Device) -> Result<()> {
    let floats = Tensor::new(&[1f32, 2.], device)?;

    for result in [
        floats.remainder(&floats),
        floats.bitwise_and(&floats),
        floats.bitwise_or(&floats),
        floats.bitwise_xor(&floats),
        floats.bitwise_left_shift(&floats),
        floats.bitwise_right_shift(&floats),
    ] {
        let message = refusal(result);
        assert!(message.contains("unsupported dtype F32"), "{message}");
    }

    // The two operands have one element type.
    let wide = Tensor::new(&[1i64, 2], device)?;
    let narrow = Tensor::new(&[1u8, 2], device)?;
    assert!(wide.bitwise_and(&narrow).is_err());
    Ok(())
}

fn a_zero_divisor_is_an_error(device: &Device) -> Result<()> {
    let dividend = Tensor::new(&[[7i64, 8], [9, 10]], device)?;
    let divisor = Tensor::new(&[[1i64, 2], [0, 4]], device)?;

    for result in [dividend.div(&divisor), dividend.remainder(&divisor)] {
        let message = refusal(result);
        assert!(message.contains("division by zero"), "{message}");
    }

    // The zero is outside what the divisor's view reads.
    let nonzero = divisor.narrow(0, 0, 1)?;
    assert_eq!(
        dividend.narrow(0, 1, 1)?.div(&nonzero)?.to_vec2::<i64>()?,
        [[9, 5]]
    );
    assert_eq!(
        dividend
            .narrow(0, 1, 1)?
            .remainder(&nonzero)?
            .to_vec2::<i64>()?,
        [[0, 0]]
    );
    // The same through a view that is not contiguous: the second column.
    assert_eq!(
        dividend
            .narrow(1, 0, 1)?
            .div(&divisor.narrow(1, 1, 1)?)?
            .to_vec2::<i64>()?,
        [[3], [2]]
    );

    // A float quotient by zero is a float.
    let floats = Tensor::new(&[1f32, -1.], device)?;
    assert_eq!(
        (&floats / &floats.zeros_like()?)?.to_vec1::<f32>()?,
        [f32::INFINITY, f32::NEG_INFINITY]
    );
    Ok(())
}

fn a_scalar_operand_is_exact_on_integers(device: &Device) -> Result<()> {
    // Values past 2^53, which no f64 arithmetic keeps.
    let large = (1i64 << 53) + 1;
    let x = Tensor::new(&[large, -large, i64::MAX], device)?;
    assert_eq!(
        (&x - 1.0)?.to_vec1::<i64>()?,
        [large - 1, -large - 1, i64::MAX - 1]
    );
    assert_eq!(
        (&x + 1.0)?.to_vec1::<i64>()?,
        [large + 1, -large + 1, i64::MIN]
    );
    assert_eq!((&x * 2.0)?.to_vec1::<i64>()?, [2 * large, -2 * large, -2]);
    assert_eq!(
        (&x / 2.0)?.to_vec1::<i64>()?,
        [large / 2, -large / 2, i64::MAX / 2]
    );
    assert_eq!(
        (1.0 - &x)?.to_vec1::<i64>()?,
        [1 - large, 1 + large, 1 - i64::MAX]
    );
    assert_eq!((1.0 + &x)?.to_vec1::<i64>()?, (&x + 1.0)?.to_vec1::<i64>()?);
    assert_eq!((2.0 * &x)?.to_vec1::<i64>()?, (&x * 2.0)?.to_vec1::<i64>()?);

    let x = Tensor::new(&[4294967295u32, 0, 7], device)?;
    assert_eq!((&x - 1.0)?.to_vec1::<u32>()?, [4294967294, 4294967295, 6]);
    assert_eq!((&x / 2.0)?.to_vec1::<u32>()?, [2147483647, 0, 3]);
    assert_eq!((1.0 - &x)?.to_vec1::<u32>()?, [2, 1, 4294967290]);

    // The scalar divided by each element, truncated toward zero.
    let x = Tensor::new(&[2i32, -2, 8], device)?;
    assert_eq!((7.0 / &x)?.to_vec1::<i32>()?, [3, -3, 0]);
    assert_eq!((&x / -2.0)?.to_vec1::<i32>()?, [-1, 1, -4]);

    // An owned tensor computes as a borrowed one does.
    assert_eq!((x.clone() + 1.0)?.to_vec1::<i32>()?, [3, -1, 9]);
    assert_eq!((1.0 - x.clone())?.to_vec1::<i32>()?, [-1, 3, -7]);
    assert_eq!((8.0 / x)?.to_vec1::<i32>()?, [4, -4, 1]);
    Ok(())
}

fn a_scalar_that_is_no_value_of_the_element_type_is_an_error(device: &Device) -> Result<()> {
    let bytes = Tensor::new(&[1u8, 2], device)?;
    let wide = Tensor::new(&[1i64, 2], device)?;

    for result in [
        // A fractional scalar: the result would be no integer.
        &bytes * 0.5,
        &wide / 0.5,
        0.5 - &wide,
        // Outside the element type's range.
        &bytes + 256.0,
        &bytes + -1.0,
        &wide + 9223372036854775808.0,
        // Not a number at all.
        &wide + f64::NAN,
        &wide * f64::INFINITY,
    ] {
        let message = refusal(result);
        assert!(message.contains("is no"), "{message}");
    }

    let message = refusal(&wide / 0.0);
    assert!(message.contains("division by zero"), "{message}");

    // The bounds themselves are values.
    assert_eq!((&bytes + 255.0)?.to_vec1::<u8>()?, [0, 1]);
    assert_eq!(
        (&wide + -9223372036854775808.0)?.to_vec1::<i64>()?,
        [i64::MIN + 1, i64::MIN + 2]
    );

    // A float tensor takes any scalar.
    let floats = bytes.to_dtype(DType::F32)?;
    assert_eq!((&floats * 0.5)?.to_vec1::<f32>()?, [0.5, 1.]);
    Ok(())
}

test_device!(arithmetic_wraps_and_the_quotient_truncates);
test_device!(remainder_takes_the_sign_of_the_divisor);
test_device!(bits_combine_and_shift);
test_device!(an_integer_operation_on_floats_is_an_error);
test_device!(a_zero_divisor_is_an_error);
test_device!(a_scalar_operand_is_exact_on_integers);
test_device!(a_scalar_that_is_no_value_of_the_element_type_is_an_error);
