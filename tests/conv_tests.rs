#[macro_use]
mod common;

use anyhow::Result;
use uniserve_tensor::{test_utils, Device, IndexOp, Tensor};

/* This test is based on the following script.
import torch
torch.manual_seed(4242)

t = torch.randn((1, 4, 5))
w = torch.randn((2, 4, 3))
print(t.flatten())
print(w.flatten())
res = torch.nn.functional.conv1d(t, w)
print(res.flatten())
res = torch.nn.functional.conv1d(t, w, padding=1)
print(res.flatten())

w_t = w.transpose(0, 1)
res = torch.nn.functional.conv_transpose1d(t, w_t)
print(res.shape)
print(res)
res = torch.nn.functional.conv_transpose1d(t, w_t, groups=2)
print(res.shape)
print(res)
*/
fn conv1d(dev: &Device) -> Result<()> {
    let t = Tensor::new(
        &[
            0.4056f32, -0.8689, -0.0773, -1.5630, 1.2279, -0.9287, -1.7030, 0.1370, 0.1866, 0.4145,
            1.8025, -0.1536, 2.2013, -0.6836, 0.2477, 1.3127, -0.6957, 0.3278, -1.0124, 0.5599,
        ],
        dev,
    )?
    .reshape((1, 4, 5))?;
    let w = Tensor::new(
        &[
            -0.8404f32, -0.3490, 0.0130, 1.3123, 0.1763, -1.9249, 1.4270, 0.9421, 0.8670, -0.7181,
            -1.1111, 0.8869, -1.2429, 1.8357, 1.6052, -1.3844, 0.3951, -1.2036, 0.6686, 1.6261,
            -0.6451, -0.0840, -1.4247, 0.5512,
        ],
        dev,
    )?
    .reshape((2, 4, 3))?;
    let res = t.conv1d(&w, 0, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 2, 3]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [2.6357, -1.3336, 4.1393, -1.1784, 3.5675, 0.5069]
    );
    let res = t.conv1d(&w, /*padding*/ 1, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 2, 5]);
    // Same as pytorch default padding: use zeros.
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [2.4509, 2.6357, -1.3336, 4.1393, 0.5657, 1.8091, -1.1784, 3.5675, 0.5069, 3.3352]
    );
    let res = {
        let t = Tensor::cat(&[&t.zeros_like()?, &t, &t.zeros_like()?], 0)?;
        t.conv1d(&w, /*padding*/ 1, 1, 1, 1)?
    };
    assert_eq!(res.dims(), [3, 2, 5]);
    // Same as pytorch default padding: use zeros.
    assert_eq!(
        test_utils::to_vec1_round(&res.i(0)?.flatten_all()?, 4)?,
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    );
    assert_eq!(
        test_utils::to_vec1_round(&res.i(1)?.flatten_all()?, 4)?,
        [2.4509, 2.6357, -1.3336, 4.1393, 0.5657, 1.8091, -1.1784, 3.5675, 0.5069, 3.3352]
    );

    let w = w.transpose(0, 1)?;
    // The CPU kernels applied in the contiguous and non contiguous cases are different.
    for w in [w.clone(), w.contiguous()?] {
        let res = t.conv_transpose1d(&w, 0, 0, 1, 1, 1)?;
        assert_eq!(res.dims(), [1, 2, 7]);
        assert_eq!(
            test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
            [
                0.0699, -1.2899, 8.3018, 5.5873, 2.4572, -2.6143, -0.0706, 1.8765, 4.8318, 1.1538,
                4.7076, -5.9745, -0.8276, 1.621
            ],
        );
        let res = t.conv_transpose1d(&w, 0, 0, 1, 1, 2)?;
        assert_eq!(res.dims(), [1, 4, 7]);
        assert_eq!(
            test_utils::to_vec2_round(&res.squeeze(0)?, 4)?,
            [
                [-1.5596, -1.8099, 2.0407, 4.8764, -0.1743, -0.735, -0.7819],
                [0.7816, 3.8152, -0.5926, 2.2515, -5.1844, -0.3157, 1.4721],
                [1.6295, 0.52, 6.2611, 0.7109, 2.6315, -1.8793, 0.7113],
                [1.0949, 1.0166, 1.7464, 2.4561, -0.79, -0.5119, 0.1488]
            ]
        );
    }
    Ok(())
}

fn conv1d_small(dev: &Device) -> Result<()> {
    let t = Tensor::new(&[0.4056f32, -0.8689, -0.0773, -1.5630], dev)?.reshape((1, 1, 4))?;
    let w = Tensor::new(&[1f32, 0., 0.], dev)?.reshape((1, 1, 3))?;
    let res = t.conv1d(&w, 0, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 2]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [0.4056, -0.8689]
    );
    let res = t.conv1d(&w, /*padding*/ 1, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 4]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [0.0, 0.4056, -0.8689, -0.0773],
    );
    Ok(())
}

/* This test is based on the following script.
import torch
torch.manual_seed(4242)

t = torch.randn((1, 4, 5, 5))
w = torch.randn((2, 4, 3, 3))
print(t.flatten())
print(w.flatten())
res = torch.nn.functional.conv2d(t, w)
print(res.flatten())

w_t = w.transpose(0, 1)
res = torch.nn.functional.conv_transpose2d(t, w_t)
print(res.shape)
print(res)

res = torch.nn.functional.conv2d(t, w, dilation=2)
print(res.shape)
print(res[0])

res = torch.nn.functional.conv_transpose2d(t, w_t, dilation=2)
print(res.shape)
print(res)
*/
fn conv2d(dev: &Device) -> Result<()> {
    let t = Tensor::new(
        &[
            0.4056f32, -0.8689, -0.0773, -1.5630, -2.8012, -1.5059, 0.3972, 1.0852, 0.4997, 3.0616,
            1.6541, 0.0964, -0.8338, -1.6523, -0.8323, -0.1699, 0.0823, 0.3526, 0.6843, 0.2395,
            1.2279, -0.9287, -1.7030, 0.1370, 0.6047, 0.3770, -0.6266, 0.3529, 2.2013, -0.6836,
            0.2477, 1.3127, -0.2260, 0.2622, -1.2974, -0.8140, -0.8404, -0.3490, 0.0130, 1.3123,
            1.7569, -0.3956, -1.8255, 0.1727, -0.3538, 2.6941, 1.0529, 0.4219, -0.2071, 1.1586,
            0.4717, 0.3865, -0.5690, -0.5010, -0.1310, 0.7796, 0.6630, -0.2021, 2.6090, 0.2049,
            0.6466, -0.5042, -0.0603, -1.6538, -1.2429, 1.8357, 1.6052, -1.3844, 0.3323, -1.3712,
            0.9634, -0.4799, -0.6451, -0.0840, -1.4247, 0.5512, -0.1747, -0.5509, -0.3742, 0.3790,
            -0.4431, -0.4720, -0.7890, 0.2620, 0.7875, 0.5377, -0.6779, -0.8088, 1.9098, 1.2006,
            -0.8, -0.4983, 1.5480, 0.8265, -0.1025, 0.5138, 0.5748, 0.3821, -0.4607, 0.0085,
        ],
        dev,
    )?;
    let w = Tensor::new(
        &[
            -0.9325f32, 0.6451, -0.8537, 0.2378, 0.8764, -0.1832, 0.2987, -0.6488, -0.2273,
            -2.4184, -0.1192, -0.4821, -0.5079, -0.5766, -2.4729, 1.6734, 0.4558, 0.2851, 1.1514,
            -0.9013, 1.0662, -0.1817, -0.0259, 0.1709, 0.5367, 0.7513, 0.8086, -2.2586, -0.5027,
            0.9141, -1.3086, -1.3343, -1.5669, -0.1657, 0.7958, 0.1432, 0.3896, -0.4501, 0.1667,
            0.0714, -0.0952, 1.2970, -0.1674, -0.3178, 1.0677, 0.3060, 0.7080, 0.1914, 1.1679,
            -0.3602, 1.9265, -1.8626, -0.5112, -0.0982, 0.2621, 0.6565, 0.5908, 1.0089, -0.1646,
            1.8032, -0.6286, 0.2016, -0.3370, 1.2555, 0.8009, -0.6488, -0.4652, -1.5685, 1.5860,
            0.5583, 0.4623, 0.6026,
        ],
        dev,
    )?;
    let t = t.reshape((1, 4, 5, 5))?;
    let w = w.reshape((2, 4, 3, 3))?;
    let res = t.conv2d(&w, 0, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 2, 3, 3]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [
            -4.2812, 2.0923, 5.2187, 7.5184, 0.752, -14.9426, 10.0087, 4.391, 0.2918, 1.6715,
            10.389, 3.6023, -4.2808, 0.2672, 5.3646, -5.2023, -2.1955, -9.4075
        ]
    );
    let res = {
        let t = Tensor::cat(&[&t.zeros_like()?, &t, &t.zeros_like()?], 0)?;
        t.conv2d(&w, 0, 1, 1, 1)?
    };
    assert_eq!(res.dims(), [3, 2, 3, 3]);
    assert_eq!(
        test_utils::to_vec1_round(&res.i(0)?.flatten_all()?, 4)?,
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    );
    assert_eq!(
        test_utils::to_vec1_round(&res.i(1)?.flatten_all()?, 4)?,
        [
            -4.2812, 2.0923, 5.2187, 7.5184, 0.752, -14.9426, 10.0087, 4.391, 0.2918, 1.6715,
            10.389, 3.6023, -4.2808, 0.2672, 5.3646, -5.2023, -2.1955, -9.4075
        ]
    );

    let res = t.conv_transpose2d(&w.transpose(0, 1)?, 0, 0, 1, 1)?;

    assert_eq!(res.dims(), [1, 2, 7, 7]);
    assert_eq!(
        test_utils::to_vec3_round(&res.i(0)?, 4)?,
        [
            [
                [-1.9918, 2.6797, -0.4599, -1.6037, 1.4131, -2.4012, 2.9277],
                [1.8016, -3.5361, 1.0757, 3.5395, -8.2168, -3.2023, 0.5375],
                [0.8243, 1.8675, 7.8929, -4.0746, -6.4415, 5.1139, 1.6889],
                [0.2722, 8.9679, 3.3477, 1.8514, -4.2896, -3.8228, -7.5632],
                [-8.5412, -5.8142, -7.1587, -1.6095, 0.4651, 0.2748, -2.0985],
                [2.0833, -0.6482, -12.1692, -4.1284, -2.9765, -0.0656, -4.5114],
                [5.307, 2.6957, 2.3087, 1.0478, 0.7808, -1.1519, -0.9579]
            ],
            [
                [1.089, 0.1872, -0.6408, -0.9897, 0.8503, 1.1019, -0.9211],
                [-0.1741, -0.2915, 4.2472, 1.9417, 1.65, 0.6303, -4.7131],
                [1.6555, 2.4026, -2.9293, 2.9953, 0.5328, 3.5873, -0.9621],
                [-1.4289, -3.2787, 4.1747, -6.0341, -4.6341, -5.7945, 4.142],
                [7.5973, 6.4431, 5.9872, 2.1639, -8.6566, 3.3143, -3.4059],
                [-0.8775, -3.048, 11.6543, 0.6442, 2.3218, -0.4765, 1.1516],
                [-5.5423, -2.5188, 1.0754, -0.0563, -2.9386, -1.1504, 1.0171]
            ]
        ]
    );

    // Dilations.
    let res = t.conv2d(&w, 0, 1, 2, 1)?;
    assert_eq!(res.dims(), [1, 2, 1, 1]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [2.45, -2.3504],
    );

    // Transpose and dilations.
    let res = t.conv_transpose2d(&w.transpose(0, 1)?, 0, 0, 1, 2)?;
    assert_eq!(res.dims(), [1, 2, 9, 9]);
    assert_eq!(
        test_utils::to_vec3_round(&res.i(0)?, 4)?,
        [
            [
                [-1.9918, 3.1652, -0.6778, -4.3442, 4.4351, 0.6652, -3.0124, -0.6031, 2.9277],
                [2.7036, -1.7156, -0.3969, 1.0516, 1.6381, -2.8886, -0.205, 2.4682, -1.0499],
                [-0.9459, 3.1631, 3.707, -4.8369, -8.5166, -1.4496, -2.7559, -3.2698, 1.4376],
                [-0.2157, 3.7786, -2.0252, -4.2633, 3.6731, -1.5142, 5.9391, -0.2622, -0.141],
                [-6.8121, -3.1744, 1.5945, 3.0637, -9.6088, 1.4446, 2.9489, -3.0082, -7.3822],
                [0.2371, 3.3303, 0.3861, 2.2646, -4.6784, 4.1235, -0.0109, 0.3176, -0.03],
                [-2.5339, -2.9564, -3.4518, -4.4594, -9.1873, -1.9709, -0.4676, 0.51, -3.5024],
                [4.007, 0.3067, -2.2954, 1.1105, -0.1992, 1.6372, -2.9268, 0.2807, -1.2787],
                [5.307, 1.1317, 1.3518, 0.9049, 3.8116, -0.4075, -0.8874, -0.2241, -0.9579]
            ],
            [
                [1.089, -0.6483, 0.0726, -0.4752, -1.3283, 1.7103, 1.0703, 0.1076, -0.9211],
                [-0.8629, 0.1376, 0.3202, 2.0955, 0.9696, 2.8988, -1.0012, 1.5049, -0.1278],
                [1.9286, -1.5255, -2.9563, 2.4589, 3.3611, -0.6951, 0.3525, -1.7724, -5.9861],
                [1.1226, 2.1561, 3.6417, 4.7546, -0.692, 4.4126, -5.1902, 6.0805, 2.3185],
                [1.0111, 0.3604, 0.6432, -3.6605, 7.9517, -9.2955, -5.2988, -3.7803, -2.0642],
                [3.3172, -1.7967, -3.6576, -2.0942, 1.3158, 0.112, -1.7405, 2.9167, 0.7957],
                [5.1001, 1.8995, -1.8639, 1.1262, 9.9629, 2.683, -3.6319, -1.1607, 0.5856],
                [-4.8445, -0.5642, 4.2317, 0.0856, 1.2267, -0.5712, 1.736, 1.0997, 0.6908],
                [-5.5423, -1.1831, -1.2176, 0.0843, 0.0446, -0.7545, -2.4798, -0.0827, 1.0171]
            ]
        ]
    );

    Ok(())
}

/* This test is based on the following script.
import torch
torch.manual_seed(4242)

t = torch.randn((1, 2, 3, 3))
w = torch.randn((1, 2, 1, 1))
print(t.flatten())
print(w.flatten())
res = torch.nn.functional.conv2d(t, w)
print(res.flatten())

w_t = w.transpose(0, 1)
res = torch.nn.functional.conv_transpose2d(t, w_t)
print(res.shape)
print(res.flatten())

t_t = w.transpose(0, 1)
res = torch.nn.functional.conv_transpose2d(t_t, w)
print(res.shape)
print(res.flatten())
*/
fn conv2d_small(dev: &Device) -> Result<()> {
    let t = Tensor::new(
        &[
            0.4056f32, -0.8689, 0.6843, 0.2395, 1.2279, -0.9287, -1.7030, 0.1370, 0.1866, 0.4145,
            -0.6266, 0.3529, 2.2013, -0.6836, 0.2477, 1.3127, -0.6957, 0.3278,
        ],
        dev,
    )?;
    let w = Tensor::new(&[-0.9259f32, 1.3017], dev)?;
    let t = t.reshape((1, 2, 3, 3))?;
    let w = w.reshape((1, 2, 1, 1))?;
    let res = t.conv2d(&w, 0, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 3, 3]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [0.164, -0.0111, -0.1742, 2.6437, -2.0268, 1.1823, 3.2855, -1.0324, 0.2539]
    );
    let res = t.conv2d(&w, 2, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 7, 7]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1640,
            -0.0111, -0.1742, 0.0, 0.0, 0.0, 0.0, 2.6437, -2.0268, 1.1823, 0.0, 0.0, 0.0, 0.0,
            3.2855, -1.0324, 0.2539, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0
        ]
    );

    let res = t.conv_transpose2d(&w.transpose(0, 1)?, 0, 0, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 3, 3]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [0.164, -0.0111, -0.1742, 2.6437, -2.0268, 1.1823, 3.2855, -1.0324, 0.2539],
    );
    let res = t.transpose(0, 1)?.conv_transpose2d(&w, 0, 0, 1, 1)?;
    assert_eq!(res.dims(), [2, 2, 3, 3]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [
            -0.3755, 0.8045, -0.6336, -0.2218, -1.1369, 0.8599, 1.5768, -0.1268, -0.1728, 0.528,
            -1.131, 0.8908, 0.3118, 1.5984, -1.2089, -2.2168, 0.1783, 0.2429, -0.3838, 0.5802,
            -0.3268, -2.0382, 0.6329, -0.2293, -1.2154, 0.6441, -0.3035, 0.5396, -0.8156, 0.4594,
            2.8654, -0.8898, 0.3224, 1.7087, -0.9056, 0.4267
        ]
    );
    Ok(())
}

fn conv2d_smaller(dev: &Device) -> Result<()> {
    let t = Tensor::new(
        &[
            0.4056f32, -0.8689, 0.6843, 0.2395, 1.2279, -0.9287, -1.7030, 0.1370, 0.1866,
        ],
        dev,
    )?;
    let w = Tensor::new(&[1f32, 1., 1., 1., 1., 1., 1., 1., 1.], dev)?;
    let t = t.reshape((1, 1, 3, 3))?;
    let w = w.reshape((1, 1, 3, 3))?;
    let res = t.conv2d(&w, 0, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 1, 1]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [-0.6197]
    );
    Ok(())
}

/* This test is based on the following script.
import torch
torch.manual_seed(4242)

t = torch.randn((1, 2, 4, 2))
w = torch.randn((1, 2, 1, 1))
print(t.flatten())
print(w.flatten())
res = torch.nn.functional.conv2d(t, w)
print(res.flatten())
*/
fn conv2d_non_square(dev: &Device) -> Result<()> {
    let t = Tensor::new(
        &[
            0.4056f32, -0.8689, -0.0773, -1.5630, -2.8012, -1.5059, 0.3972, 1.0852, 0.4997, 3.0616,
            1.6541, 0.0964, -0.8338, -1.6523, -0.8323, -0.1699,
        ],
        dev,
    )?;
    let w = Tensor::new(&[-1.1351f32, 1.3841], dev)?;
    let t = t.reshape((1, 2, 4, 2))?;
    let w = w.reshape((1, 2, 1, 1))?;
    let res = t.conv2d(&w, 0, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 1, 4, 2]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        [0.2312, 5.2238, 2.3772, 1.9076, 2.0256, -0.5776, -1.6028, -1.467]
    );
    Ok(())
}

/*
import torch
torch.manual_seed(4242)

t = torch.randn((1, 4, 5, 5))
w = torch.randn((2, 4, 3, 3))
print(t.flatten())
print(w.flatten())
res = torch.nn.functional.conv2d(t, w)
print(res.flatten())
loss = (res ** 2).sum()
print(loss)

res = torch.nn.functional.conv2d(t, w, stride=2)
print(res.flatten())
loss = (res ** 2).sum()
print(loss)
*/
fn conv2d_sum_of_squares(dev: &Device) -> Result<()> {
    let t = Tensor::from_slice(
        &[
            0.4056f32, -0.8689, -0.0773, -1.5630, -2.8012, -1.5059, 0.3972, 1.0852, 0.4997, 3.0616,
            1.6541, 0.0964, -0.8338, -1.6523, -0.8323, -0.1699, 0.0823, 0.3526, 0.6843, 0.2395,
            1.2279, -0.9287, -1.7030, 0.1370, 0.6047, 0.3770, -0.6266, 0.3529, 2.2013, -0.6836,
            0.2477, 1.3127, -0.2260, 0.2622, -1.2974, -0.8140, -0.8404, -0.3490, 0.0130, 1.3123,
            1.7569, -0.3956, -1.8255, 0.1727, -0.3538, 2.6941, 1.0529, 0.4219, -0.2071, 1.1586,
            0.4717, 0.3865, -0.5690, -0.5010, -0.1310, 0.7796, 0.6630, -0.2021, 2.6090, 0.2049,
            0.6466, -0.5042, -0.0603, -1.6538, -1.2429, 1.8357, 1.6052, -1.3844, 0.3323, -1.3712,
            0.9634, -0.4799, -0.6451, -0.0840, -1.4247, 0.5512, -0.1747, -0.5509, -0.3742, 0.3790,
            -0.4431, -0.4720, -0.7890, 0.2620, 0.7875, 0.5377, -0.6779, -0.8088, 1.9098, 1.2006,
            -0.8, -0.4983, 1.5480, 0.8265, -0.1025, 0.5138, 0.5748, 0.3821, -0.4607, 0.0085,
        ],
        (1, 4, 5, 5),
        dev,
    )?;
    let w = Tensor::from_slice(
        &[
            -0.9325f32, 0.6451, -0.8537, 0.2378, 0.8764, -0.1832, 0.2987, -0.6488, -0.2273,
            -2.4184, -0.1192, -0.4821, -0.5079, -0.5766, -2.4729, 1.6734, 0.4558, 0.2851, 1.1514,
            -0.9013, 1.0662, -0.1817, -0.0259, 0.1709, 0.5367, 0.7513, 0.8086, -2.2586, -0.5027,
            0.9141, -1.3086, -1.3343, -1.5669, -0.1657, 0.7958, 0.1432, 0.3896, -0.4501, 0.1667,
            0.0714, -0.0952, 1.2970, -0.1674, -0.3178, 1.0677, 0.3060, 0.7080, 0.1914, 1.1679,
            -0.3602, 1.9265, -1.8626, -0.5112, -0.0982, 0.2621, 0.6565, 0.5908, 1.0089, -0.1646,
            1.8032, -0.6286, 0.2016, -0.3370, 1.2555, 0.8009, -0.6488, -0.4652, -1.5685, 1.5860,
            0.5583, 0.4623, 0.6026,
        ],
        (2, 4, 3, 3),
        dev,
    )?;
    let res = t.conv2d(&w, 0, 1, 1, 1)?;
    let loss = res.sqr()?.sum_all()?;
    assert_eq!(test_utils::to_vec0_round(&loss, 2)?, 741.12f32);

    // Same as before but with stride.
    let res = t.conv2d(&w, 0, 2, 1, 1)?;
    let loss = res.sqr()?.sum_all()?;
    assert_eq!(test_utils::to_vec0_round(&loss, 2)?, 277.16f32);

    // Replicate the issue from https://github.com/huggingface/candle/issues/1212
    let res = t.i((.., .., 0..4, 0..4))?.conv2d(&w, 0, 2, 1, 1)?;
    let loss = res.sqr()?.sum_all()?;
    assert_eq!(test_utils::to_vec0_round(&loss, 2)?, 21.12f32);

    // Conv Transpose 2d Test
    //tested against following python

    // import torch
    // torch.manual_seed(4242)
    // padding = 4
    // outpadding = 2
    // dilation = 3
    // stride = 3
    // input = torch.randn((1, 4, 7, 5))
    // kernel = torch.randn((4, 2, 3, 5))
    // print("input", input.flatten())
    // print("kernel", kernel.flatten())
    // res = torch.nn.functional.conv_transpose2d(
    //     input,
    //     kernel,
    //     stride=stride,
    //     padding=padding,
    //     dilation=dilation,
    //     output_padding=outpadding,
    // )
    // print(res.shape)
    // loss = (res**2).sum()
    // print(loss)

    let padding = 4;
    let outpadding = 2;
    let dilation = 3;
    let stride = 3;

    let t = Tensor::from_slice(
        &[
            0.4056_f32, -0.8689, -0.0773, -1.5630, -2.8012, -1.5059, 0.3972, 1.0852, 0.4997,
            3.0616, 1.6541, 0.0964, -0.8338, -1.6523, -0.8323, -0.1699, 0.0823, 0.3526, 0.6843,
            0.2395, 1.2279, -0.9287, -1.7030, 0.1370, 0.6047, 0.3770, -0.6266, 0.3529, 2.2013,
            -0.6836, 0.2477, 1.3127, -0.2260, 0.2622, -1.2974, -0.8140, -0.8404, -0.3490, 0.0130,
            1.3123, 1.7569, -0.3956, -1.8255, 0.1727, -0.3538, 2.6941, 1.0529, 0.4219, -0.2071,
            1.1586, 0.4717, 0.3865, -0.5690, -0.5010, -0.1310, 0.7796, 0.6630, -0.2021, 2.6090,
            0.2049, 0.6466, -0.5042, -0.0603, -1.6538, -1.2429, 1.8357, 1.6052, -1.3844, 0.3323,
            -1.3712, 0.9634, -0.4799, -0.6451, -0.0840, -1.4247, 0.5512, -0.1747, -0.5509, -0.3742,
            0.3790, -0.4431, -0.4720, -0.7890, 0.2620, 0.5411, -1.1715, -2.4997, 2.3249, -0.8912,
            -0.4733, -0.5701, -2.8888, -1.4112, -0.5471, -0.9234, -1.1660, 0.4189, -0.7465,
            -0.6473, 0.1402, 0.7875, 0.5377, -0.6779, -0.8088, -0.4864, -0.2312, 0.9279, 0.1264,
            1.5480, 0.8265, -0.1025, 0.5138, -0.2512, 0.1576, 1.2705, 0.3641, -0.9325, 0.6451,
            -0.8537, 0.2378, 0.1794, 0.2752, -0.3687, -1.1149, -0.1410, -0.5829, -0.0892, 1.4258,
            -2.2789, 0.5270, 0.1825, 1.7007, -0.5263, -0.2954, 0.4440, 0.5537, 0.3492, 0.6186,
            1.6475, 0.2219,
        ],
        (1, 4, 7, 5),
        dev,
    )?;

    #[rustfmt::skip]
    let w = Tensor::from_slice(
        &[
            -1.1744_f32, 0.3266, 2.5893, 1.0142, 0.1763, 0.7752, 0.6604, 0.2029, -0.2145, 0.7234,
            -0.3441, -1.5400, -0.6333, 0.6613, 0.2083, 0.6230, -1.7002, 0.3393, 0.4049, 1.0762,
            0.2723, 1.4181, 0.0029, -0.2122, 1.7668, 1.4168, 0.3320, -0.2719, 0.7932, -0.7204,
            0.4447, 0.1211, 0.5908, 1.0089, -0.1646, 1.8033, -0.6286, 0.2016, -0.3370, 1.2555,
            0.8009, -0.6488, -0.4652, -1.5685, 1.5860, 0.5583, 0.4623, 0.6026, 0.8828, 2.4990,
            0.6811, -0.3369, 1.3320, 1.7669, -1.1067, 1.2958, -0.9415, -0.9655, -0.4462, 0.7181,
            0.5181, -1.1658, -1.8467, -0.7763, 1.2769, 0.8651, 0.9890, 1.5092, 0.7207, -0.8481,
            0.7417, 0.3375, -1.2685, 1.4572, 1.0915, 0.1093, -0.8550, -0.5831, -0.6309, -0.2509,
            0.5220, -0.0914, 0.7900, 0.1096, 0.3258, 0.2723, -1.0942, -0.3393, -0.1653, 0.5732,
            -0.8014, 1.8194, -1.9023, 0.2127, 1.8636, -0.8979, 0.1927, -0.2778, 0.3105, 0.0071,
            -1.1823, 0.2476, -0.7178, -1.3821, 1.0769, -0.4376, -0.9967, -0.1227, 1.6197, -1.0604,
            0.1372, 0.8141, -0.6163, 0.7304, -0.8285, 2.0636, -0.7176, 0.2495, -0.2581, -0.4478,
        ],
        (4, 2, 3, 5),
        dev,
    )?;
    let res = t.conv_transpose2d(&w, padding, outpadding, stride, dilation)?;
    let loss = res.sqr()?.sum_all()?;
    assert_eq!(test_utils::to_vec0_round(&loss, 0)?, 2904.0);

    // Test the same, but then with the following properties, t & w are unmodified.
    let padding = 1;
    let outpadding = 1;
    let dilation = 1;
    let stride = 2;

    let res = t.conv_transpose2d(&w, padding, outpadding, stride, dilation)?;
    let loss = res.sqr()?.sum_all()?;
    assert_eq!(test_utils::to_vec0_round(&loss, 0)?, 3627.0); // torch gives 3626.8560

    Ok(())
}

/* Regression test for https://github.com/huggingface/candle/issues/3404

When C_in == H == W, NCHW and NHWC strides are numerically identical:
  NCHW [B, C, H, W] strides: [C*H*W, H*W, W, 1]
  NHWC [B, H, W, C] strides: [H*W*C, W*C, C, 1]
  When C == H == W: both = [C^3, C^2, C, 1]

conv2d_tiled used stride comparison to skip NCHW→NHWC conversion, causing it
to read NCHW data as NHWC when C == H == W, producing completely wrong results.

Reference values from PyTorch:
  import torch
  torch.manual_seed(42)
  t = torch.randn(1, 4, 4, 4)
  w = torch.randn(8, 4, 3, 3)
  b = torch.randn(8)
  res = torch.nn.functional.conv2d(t, w, b, padding=1)
*/
fn conv2d_c_eq_h_eq_w(dev: &Device) -> Result<()> {
    let t = Tensor::new(
        &[
            1.9269f32, 1.4873, 0.9007, -2.1055, 0.6784, -1.2345, -0.0431, -1.6047, -0.7521, 1.6487,
            -0.3925, -1.4036, -0.7279, -0.5594, -0.7688, 0.7624, 1.6423, -0.1596, -0.4974, 0.4396,
            -0.7581, 1.0783, 0.8008, 1.6806, 1.2791, 1.2964, 0.6105, 1.3347, -0.2316, 0.0418,
            -0.2516, 0.8599, -1.3847, -0.8712, -0.2234, 1.7174, 0.3189, -0.4245, 0.3057, -0.7746,
            -1.5576, 0.9956, -0.8798, -0.6011, -1.2742, 2.1228, -1.2347, -0.4879, -0.9138, -0.6581,
            0.0780, 0.5258, -0.4880, 1.1914, -0.8140, -0.7360, -1.4032, 0.0360, -0.0635, 0.6756,
            -0.0978, 1.8446, -1.1845, 1.3835,
        ],
        dev,
    )?;
    let t = t.reshape((1, 4, 4, 4))?;

    // Use a simple weight: all ones kernel for easy verification
    let w = Tensor::ones((2, 4, 3, 3), uniserve_tensor::DType::F32, dev)?;
    let res = t.conv2d(&w, 1, 1, 1, 1)?;
    assert_eq!(res.dims(), [1, 2, 4, 4]);

    // With all-ones kernel, each output is sum of 4*3*3=36 input values in the receptive field.
    // The result for both output channels should be identical (same kernel).
    // Verify channel 0 == channel 1
    let c0 = res.i((.., 0..1, .., ..))?;
    let c1 = res.i((.., 1..2, .., ..))?;
    let diff = c0
        .sub(&c1)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff < 1e-6,
        "conv2d with C==H==W: output channels should be identical for all-ones kernel, got diff={diff}"
    );

    // Verify a specific output value.
    // With padding=1, the top-left output sums the 2x2 visible region across all 4 channels.
    // t[0,:,0:2,0:2] with all-ones kernel:
    // ch0: 1.9269+1.4873+0.6784+(-1.2345) = 2.8581
    // ch1: 1.6423+(-0.1596)+(-0.7581)+1.0783 = 1.8029
    // ch2: (-1.3847)+(-0.8712)+0.3189+(-0.4245) = -2.3615
    // ch3: (-0.9138)+(-0.6581)+(-0.4880)+1.1914 = -0.8685
    // total = 1.4310
    let actual_corner: f32 = res.i((0, 0, 0, 0))?.to_scalar()?;
    let corner_diff = (actual_corner - 1.4310).abs();
    assert!(
        corner_diff < 1e-3,
        "conv2d with C==H==W: top-left corner mismatch: got {actual_corner}, expected 1.4310, diff={corner_diff}"
    );

    Ok(())
}

// Regression test for https://github.com/huggingface/candle/issues/3735: a non-contiguous
// kernel and its contiguous copy must produce identical results.
fn conv2d_noncontiguous_kernel(dev: &Device) -> Result<()> {
    let input_data: Vec<f32> = (1..=32).map(|v| v as f32 * 0.1).collect();
    let weight_data: Vec<f32> = (1..=12).map(|v| v as f32 * 0.1).collect();
    let input = Tensor::from_slice(&input_data, (1, 2, 4, 4), dev)?;
    let weight = Tensor::from_slice(&weight_data, (3, 4, 1, 1), dev)?;

    // Non-contiguous kernel: a channel slice of a larger weight, so it has a non-zero offset.
    let kernel = weight.i((.., 1..3, .., ..))?;
    let res = input.conv2d(&kernel, 0, 2, 1, 1)?;

    // Reference: the same logical kernel forced contiguous.
    let kernel_ref = kernel.contiguous()?;
    let res_ref = input.conv2d(&kernel_ref, 0, 2, 1, 1)?;

    assert_eq!(res.dims(), [1, 3, 2, 2]);
    assert_eq!(
        test_utils::to_vec1_round(&res.flatten_all()?, 4)?,
        test_utils::to_vec1_round(&res_ref.flatten_all()?, 4)?,
    );
    Ok(())
}

test_device!(conv1d);
test_device!(conv1d_small);
test_device!(conv2d);
test_device!(conv2d_non_square);
test_device!(conv2d_small);
test_device!(conv2d_smaller);
test_device!(conv2d_sum_of_squares);
test_device!(conv2d_c_eq_h_eq_w);
test_device!(conv2d_noncontiguous_kernel);
