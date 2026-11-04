//! Prints, descriptor by descriptor, where cuDNN refuses a convolution.
//!
//! `Cudnn::plan` reports the call that failed (`cudnnBackendFinalize`) and
//! cuDNN's own text for it, which on this library is the same sentence for
//! every descriptor: the attribute or the description cuDNN disliked is only
//! in its log. So this program builds the convolutions the models use one
//! step at a time and prints what cuDNN accepted, then tries the two
//! deprecated handle attributes cuDNN 9.8 dropped, which tells a library that
//! still requires them from a description it refuses.
//!
//! Run it on the machine under test, with cuDNN's log on, which names the
//! attribute:
//!
//! ```text
//! CUDNN_LOGLEVEL_DBG=3 CUDNN_LOGDEST_DBG=stderr cargo run -p platform --example cudnn_probe
//! ```
//!
//! It exits 1 when any step was refused, so a runner's summary shows it.

use platform::{Convolution, Cudnn, DataType, Device, DeviceId, Error, Tensor};

fn main() {
    if let Err(error) = run() {
        eprintln!("the probe could not start: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Error> {
    let device = Device::new(DeviceId(0))?;
    let cudnn = Cudnn::new(&device)?;
    let version = cudnn.version();
    println!(
        "cuDNN {}.{}.{}",
        version / 10000,
        version / 100 % 100,
        version % 100
    );

    // The audio decoder's convolution: rank three, filter (12, 6, 7), the
    // layer's own padding 1, stride 1.
    let audio = Convolution::new(
        Tensor::contiguous(vec![1, 6, 512], DataType::F32),
        Tensor::contiguous(vec![12, 6, 7], DataType::F32),
        Tensor::contiguous(vec![1, 12, 508], DataType::F32),
        vec![1],
        vec![1],
        vec![1],
    );
    // The same decoder's upsampler: ConvTranspose1d(2, 3, 4, stride 2,
    // padding 1), which is the backward data pass of the forward convolution
    // with swapped batch channels.
    let transposed = Convolution::transpose(
        Tensor::contiguous(vec![1, 2, 4], DataType::F32),
        Tensor::contiguous(vec![2, 3, 4], DataType::F32),
        Tensor::contiguous(vec![1, 3, 8], DataType::F32),
        vec![1],
        vec![2],
        vec![1],
        vec![0],
    );
    // The video VAE's block: rank five, a 3x3x3 kernel, padding 1.
    let video = Convolution::new(
        Tensor::contiguous(vec![1, 2, 2, 4, 4], DataType::F32),
        Tensor::contiguous(vec![3, 2, 3, 3, 3], DataType::F32),
        Tensor::contiguous(vec![1, 3, 2, 4, 4], DataType::F32),
        vec![1, 1, 1],
        vec![1, 1, 1],
        vec![1, 1, 1],
    );

    let mut refused = 0usize;
    for (name, convolution) in [
        ("audio decoder, 1-D forward", &audio),
        ("audio decoder, 1-D transposed", &transposed),
        ("video VAE, 3-D forward", &video),
    ] {
        println!("\n== {name}");
        for (step, failure) in cudnn.probe(convolution) {
            match failure {
                None => println!("   ok   {step}"),
                Some(message) => {
                    refused += 1;
                    println!("   FAIL {step}");
                    println!("        {message}");
                }
            }
        }
    }

    if refused > 0 {
        eprintln!("\n{refused} step(s) refused; cuDNN's log above names the attribute");
        std::process::exit(1);
    }
    println!("\nevery step accepted");
    Ok(())
}
