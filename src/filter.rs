use image::Image;
use pixel::Pixel;
use convolution::{apply_convolution, ConvolutionMatrix};

pub enum FilterType {
    MirrorX,
    MirrorY,
    Grayscale,
    Invert,
    Convolution(ConvolutionMatrix),
    EdgeDetection,
    SobelFilter(u8),
}

pub trait ImageFilterExt {
    fn filter(&mut self, FilterType);
}

impl<'a> ImageFilterExt for Image<'a> {
    fn filter(&mut self, filter: FilterType) {
        match filter {
            FilterType::MirrorX => mirror_x(self),
            FilterType::MirrorY => mirror_y(self),
            FilterType::Grayscale => grayscale(self),
            FilterType::Invert => invert(self),
            FilterType::Convolution(matrix) => convolution(self, matrix),
            FilterType::EdgeDetection => edge_detection(self),
            FilterType::SobelFilter(num) => edge(self, num),
        }
    }
}

fn mirror_x(image: &mut Image) {
    for i in 0..image.pixels.len() {
        let mid = image.width / 2;
        let (row, col) = image.index_to_row_col(i);

        if col < mid {
            let j = image.row_col_to_index(row, image.width - 1 - col);
            image.pixels[j] = image.pixels[i].clone();
        }
    }
}

fn mirror_y(image: &mut Image) {
    for i in 0..image.pixels.len() {
        let mid = image.height / 2;
        let (row, col) = image.index_to_row_col(i);

        if row < mid {
            let j = image.row_col_to_index(image.height - 1 - row, col);
            image.pixels[j] = image.pixels[i].clone();
        }
    }
}

fn grayscale(image: &mut Image) {
    for i in 0..image.pixels.len() {
        image.pixels[i].grayscale();
    }
}

fn convolution(image: &mut Image, matrix: ConvolutionMatrix) {
    let mut pixels_copy: Vec<Pixel> = image.pixels.iter().cloned().collect();
    let original = Image {
        width: image.width,
        height: image.height,
        pixels: &mut pixels_copy[..],
    };

    for i in 0..image.pixels.len() {
        let (row, col) = image.index_to_row_col(i);
        if row > 0 && row < (image.height - 1) && col > 0 && col < (image.width - 1) {  // ignore outer border
            let (red_n, green_n, blue_n) = original.get_neighbour_colours(i);
            let red = apply_convolution(red_n, matrix);
            let green = apply_convolution(green_n, matrix);
            let blue = apply_convolution(blue_n, matrix);
            image.pixels[i] = Pixel::rgb(red, green, blue);
        }
    }
}

fn invert(image: &mut Image) {
    for i in 0..image.pixels.len() {
        image.pixels[i].invert();
    }
}

fn edge( image: &mut Image, num: u8) {
    let SOBEL_1_X = vec![
        vec![-1.0, 0.0, 1.0],
        vec![-2.0, 0.0, 2.0],
        vec![-1.0, 0.0, 1.0],
    ];

    let SOBEL_1_Y = vec![
        vec![-1.0, -2.0, -1.0],
        vec![ 0.0,  0.0,  0.0],
        vec![ 1.0,  2.0,  1.0],
    ];
    
    let SOBEL_2_X = vec![
        vec![-1.0,  -2.0, 0.0,  2.0, 1.0],
        vec![-4.0, -10.0, 0.0, 10.0, 4.0],
        vec![-7.0, -17.0, 0.0, 17.0, 7.0],
        vec![-4.0, -10.0, 0.0, 10.0, 4.0],
        vec![-1.0,  -2.0, 0.0,  2.0, 1.0],
    ];
    
    let SOBEL_2_Y = vec![
        vec![-1.0,  -4.0,  -7.0,  -4.0, -1.0],
        vec![-2.0, -10.0, -17.0, -10.0, -2.0],
        vec![ 0.0,   0.0,   0.0,   0.0,  0.0],
        vec![ 2.0,  10.0,  17.0,  10.0,  2.0],
        vec![ 1.0,   4.0,   7.0,   4.0,  1.0],
    ];
    
    let SOBEL_3_X = vec![
        vec![ -1.0,  -3.0,  -3.0, 0.0,  3.0,  3.0,  1.0],
        vec![ -4.0, -11.0, -13.0, 0.0, 13.0, 11.0,  4.0],
        vec![ -9.0, -26.0, -30.0, 0.0, 30.0, 26.0,  9.0],
        vec![-13.0, -34.0, -40.0, 0.0, 40.0, 34.0, 13.0],
        vec![ -9.0, -26.0, -30.0, 0.0, 30.0, 26.0,  9.0],
        vec![ -4.0, -11.0, -13.0, 0.0, 13.0, 11.0,  4.0],
        vec![ -1.0,  -3.0,  -3.0, 0.0,  3.0,  3.0,  1.0],
    ];
    
    let SOBEL_3_Y = vec![
        vec![-1.0,  -4.0,  -9.0, -13.0,  -9.0,  -4.0, -1.0],
        vec![-3.0, -11.0, -26.0, -34.0, -26.0, -11.0, -3.0],
        vec![-3.0, -13.0, -30.0, -40.0, -30.0, -13.0, -3.0],
        vec![ 0.0,   0.0,   0.0,   0.0,   0.0,   0.0,  0.0],
        vec![ 3.0,  13.0,  30.0,  40.0,  30.0,  13.0,  3.0],
        vec![ 3.0,  11.0,  26.0,  34.0,  26.0,  11.0,  3.0],
        vec![ 1.0,   4.0,   9.0,  13.0,   9.0,   4.0,  1.0],
    ];
    let matrix_x;
    let matrix_y;
    if num == 1 {
        matrix_x = SOBEL_1_X;
        matrix_y = SOBEL_1_Y;
    } else if num == 2 {
        matrix_x = SOBEL_2_X;
        matrix_y = SOBEL_2_Y;
    } else if num == 3 {
        matrix_x = SOBEL_3_X;
        matrix_y = SOBEL_3_Y;
    } else {
        return;
    }

    // Create a copy of our Image so we do not mess up the color values mid-calculation
    let mut pixels_copy: Vec<Pixel> = image.pixels.iter().cloned().collect();
    let original = Image {
        width: image.width,
        height: image.height,
        pixels: &mut pixels_copy[..],
    };

    // Turn all pixels into grayscale
    for i in 0..image.pixels.len() {
        image.pixels[i].grayscale();
    }

    let margin = matrix_x.len() / 2;

    for i in 0..original.pixels.len() {
        let (row,column) = original.index_to_row_col(i);
        if row < margin || row > original.height - (margin+1) || column < margin || column > original.width - (margin+1) {
            continue;
        }

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;

        for j in -(margin as i32)..(margin as i32)+1 {
            for k in -(margin as i32)..(margin as i32)+1 {
                let calc_index = original.row_col_to_index((row as i32 + j) as usize, (column as i32 + k) as usize);
                sum_x += matrix_x[(j+margin as i32) as usize][(k+margin as i32) as usize] * original.pixels[calc_index].red as f64;
                sum_y += matrix_y[(j+margin as i32) as usize][(k+margin as i32) as usize] * original.pixels[calc_index].red as f64;
            }
        }

        let g = (sum_x * sum_x + sum_y * sum_y).sqrt();

        image.pixels[i].set_gray( g as u8 );
    }

}


fn edge_detection( image: &mut Image) {
    let mut pixels_copy: Vec<Pixel> = image.pixels.iter().cloned().collect();
    let original = Image {
        width: image.width,
        height: image.height,
        pixels: &mut pixels_copy[..],
    };
    // Sobel Filter:
    //       [ -1 0 1 ]      [-1 -2 -1 ]
    // Hx =  [ -2 0 2 ] Hy = [ 0  0  0 ]
    //       [ -1 0 1 ]      [ 1  2  1 ]
    let _h_x = [
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1,
    ];
    let _h_y = [
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1,
    ];



    for i in 0..image.pixels.len() {
        image.pixels[i].grayscale();
    }

    for i in 0..original.pixels.len() {
        let (row,column) = original.index_to_row_col(i);
        if row == 0 || row == original.height - 1 || column == 0 || column == original.width - 1 {
            continue;
        }
        let mut sum_x: i32 = original.pixels[i - original.width + 1].red as i32;
        sum_x += - (original.pixels[i - original.width - 1].red as i32);
        sum_x += - 2 * (original.pixels[i - 1].red as i32);
        sum_x += 2 * (original.pixels[i +1].red as i32);
        sum_x += - (original.pixels[i + original.width - 1].red as i32);
        sum_x += original.pixels[i + original.width + 1].red as i32;

        let mut sum_y: i32 = -1 * (original.pixels[i - original.width - 1].red as i32);
        sum_y += - 2 * (original.pixels[i - original.width].red as i32);
        sum_y += - (original.pixels[i - original.width + 1].red as i32);
        sum_y += original.pixels[i + original.width - 1].red as i32;
        sum_y += 2 * (original.pixels[i + original.width].red as i32);
        sum_y += original.pixels[i + original.width + 1].red as i32;

        // let g = ((sum_x * sum_x + sum_y * sum_y) as f64).sqrt();
        let g = ((sum_x * sum_x + sum_y * sum_y) as f64).sqrt();

        image.pixels[i].set_gray( g as u8 );
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_filter_mirror_x() {
        let mut pixels = [
            Pixel::rgb(100, 100, 100),
            Pixel::rgb(0, 0, 0),
            Pixel::rgb(100, 100, 100),
            Pixel::rgb(0, 0, 0),
        ];

        let mut image = Image::from_raw(&mut pixels[0], 2, 2);
        image.filter(FilterType::MirrorX);

        assert_eq!(image.pixels, [
            Pixel::rgb(100, 100, 100),
            Pixel::rgb(100, 100, 100),
            Pixel::rgb(100, 100, 100),
            Pixel::rgb(100, 100, 100),
        ]);

        let mut pixels = [
            Pixel::rgb(100, 100, 100),
            Pixel::rgb(0, 0, 0),
            Pixel::rgb(0, 0, 0),
            Pixel::rgb(100, 100, 100),
            Pixel::rgb(0, 0, 0),
            Pixel::rgb(0, 0, 0),
            Pixel::rgb(100, 100, 100),
            Pixel::rgb(0, 0, 0),
            Pixel::rgb(0, 0, 0),
        ];

        let mut image = Image::from_raw(&mut pixels[0], 3, 3);
        image.filter(FilterType::MirrorX);

        assert_eq!(image.pixels, [
            Pixel::rgb(100, 100, 100),
            Pixel::rgb(0, 0, 0),
            Pixel::rgb(100, 100, 100),
            Pixel::rgb(100, 100, 100),
            Pixel::rgb(0, 0, 0),
            Pixel::rgb(100, 100, 100),
            Pixel::rgb(100, 100, 100),
            Pixel::rgb(0, 0, 0),
            Pixel::rgb(100, 100, 100),
        ]);
    }

    #[test]
    fn test_filter_mirror_y() {
        let mut pixels = [
            Pixel::rgb(100, 100, 100),
            Pixel::rgb(100, 100, 100),
            Pixel::rgb(0, 0, 0),
            Pixel::rgb(0, 0, 0),
        ];

        let mut image = Image::from_raw(&mut pixels[0], 2, 2);
        image.filter(FilterType::MirrorY);

        assert_eq!(image.pixels, [
            Pixel::rgb(100, 100, 100),
            Pixel::rgb(100, 100, 100),
            Pixel::rgb(100, 100, 100),
            Pixel::rgb(100, 100, 100),
        ]);

        let mut pixels = [
            Pixel::rgb(100, 100, 100),
            Pixel::rgb(100, 100, 100),
            Pixel::rgb(100, 100, 100),
            Pixel::rgb(0, 0, 0),
            Pixel::rgb(0, 0, 0),
            Pixel::rgb(0, 0, 0),
            Pixel::rgb(0, 0, 0),
            Pixel::rgb(0, 0, 0),
            Pixel::rgb(0, 0, 0),
        ];

        let mut image = Image::from_raw(&mut pixels[0], 3, 3);
        image.filter(FilterType::MirrorY);

        assert_eq!(image.pixels, [
            Pixel::rgb(100, 100, 100),
            Pixel::rgb(100, 100, 100),
            Pixel::rgb(100, 100, 100),
            Pixel::rgb(0, 0, 0),
            Pixel::rgb(0, 0, 0),
            Pixel::rgb(0, 0, 0),
            Pixel::rgb(100, 100, 100),
            Pixel::rgb(100, 100, 100),
            Pixel::rgb(100, 100, 100),
        ]);
    }

    #[test]
    fn test_filter_grayscale() {
        let mut pixels = [
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
        ];

        let mut image = Image::from_raw(&mut pixels[0], 2, 2);
        image.filter(FilterType::Grayscale);

        assert_eq!(image.pixels, [
            Pixel::rgb(150, 150, 150),
            Pixel::rgb(150, 150, 150),
            Pixel::rgb(150, 150, 150),
            Pixel::rgb(150, 150, 150),
        ]);
    }

    #[test]
    fn test_convolution() {
        let mut pixels = [
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
        ];

        let mut image = Image::from_raw(&mut pixels[0], 3, 3);
        image.filter(FilterType::Convolution([
            [0.,0.,0.],
            [0.,1.,0.],         // identity matrix
            [0.,0.,0.],
        ]));

        assert_eq!(image.pixels, [
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
        ]);

        image.filter(FilterType::Convolution([
            [0.,0.,0.],
            [0.,0.,0.],         // zero out matrix
            [0.,0.,0.],
        ]));

        assert_eq!(image.pixels, [
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(0, 0, 0),
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
        ]);
    }

    #[test]
    fn test_invert() {
        let mut pixels = [
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
            Pixel::rgb(100, 150, 200),
        ];

        let mut image = Image::from_raw(&mut pixels[0], 2, 2);
        image.filter(FilterType::Invert);

        assert_eq!(image.pixels, [
            Pixel::rgb(155, 105, 55),
            Pixel::rgb(155, 105, 55),
            Pixel::rgb(155, 105, 55),
            Pixel::rgb(155, 105, 55),
        ]);
    }
    
    #[test]
    fn test_sobel() {
        let mut pixels = [
            Pixel::rgb(119,119,119),
            Pixel::rgb(80,80,80),
            Pixel::rgb(122,122,122),
            Pixel::rgb(177,177,177),
            Pixel::rgb(154,154,154),
            Pixel::rgb(212,212,212),
            Pixel::rgb(89,89,89),
            Pixel::rgb(25,25,25),
            Pixel::rgb(152,152,152),
        ];
        let mut image = Image::from_raw(&mut pixels[0], 3,3);
        image.filter(FilterType::EdgeDetection);

        assert_eq!(image.pixels[4].red, 174);
    }

    #[test]
    fn test_edge_sobel_1() {
        let mut pixels = [
            Pixel::rgb(119,119,119),
            Pixel::rgb(80,80,80),
            Pixel::rgb(122,122,122),
            Pixel::rgb(177,177,177),
            Pixel::rgb(154,154,154),
            Pixel::rgb(212,212,212),
            Pixel::rgb(89,89,89),
            Pixel::rgb(25,25,25),
            Pixel::rgb(152,152,152),
        ];
        let mut image = Image::from_raw(&mut pixels[0], 3,3);
        image.filter(FilterType::SobelFilter(1));

        assert_eq!(image.pixels[4].red, 174);
    }
}
