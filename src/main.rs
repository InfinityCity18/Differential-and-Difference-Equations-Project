use nalgebra;
use std::rc::Rc;
use std::{clone, f64::consts::PI};
use plotters::prelude::*;

const OMEGA_L: f64 = 0.0;
const OMEGA_R: f64 = 2.0;

struct C1 {
    pub d0: F, // funkcja
    pub d1: F, // jej pochodna
    pub left: f64,
    pub right: f64,
}

#[derive(Clone)]
struct F {
    f: Rc<dyn Fn(f64) -> f64>,
}

impl std::ops::Mul for F {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        F {
            f: Rc::new(move |x: f64| (self.f)(x) * (rhs.f)(x)),
        }
    }
}

impl F {
    fn f(&self, x: f64) -> f64 {
        (self.f)(x)
    }
}

fn main() {
    println!("Hello, world!");
    let f = F {
        f: Rc::new(|x: f64| x.sin()),
    };
    solve(2000);
}

fn gauss_quadrature(f: F, a: f64, b: f64) -> f64 {
    let inv_sqrt_3: f64 = 1.0 / 3f64.sqrt();
    let b_m_a = (b - a) / 2.0;
    let a_p_b = (a + b) / 2.0;

    b_m_a * (f.f(b_m_a * inv_sqrt_3 + a_p_b) + f.f(b_m_a * -inv_sqrt_3 + a_p_b))
}

#[allow(non_snake_case)]
fn E(x: f64) -> f64 {
    match x {
        _ if 0.0 <= x && x <= 1.0 => 2.0,
        _ if 1.0 < x && x <= 2.0 => 6.0,
        _ => {
            panic!("Left range of function E");
        }
    }
}

fn generate_pyramid(a: f64, b: f64) -> C1 {
    C1 {
        d0: F {
            f: Rc::new(move |x: f64| match x {
                _ if a <= x && x <= (a + b) / 2.0 => 2.0 * (x - a) / (b - a),
                _ if (a + b) / 2.0 < x && x <= b => 2.0 * (b - x) / (b - a),
                _ => 0.0,
            }),
        },
        d1: F {
            f: Rc::new(move |x: f64| match x {
                _ if a <= x && x <= (a + b) / 2.0 => 2.0 / (b - a),
                _ if (a + b) / 2.0 < x && x <= b => -2.0 / (b - a),
                _ => 0.0,
            }),
        },
        left: a,
        right: b,
    }
}

fn create_pyramids(a: f64, b: f64, n: usize) -> Vec<C1> {
    let mut pyramids: Vec<C1> = Vec::new();
    let h = (b - a) / n as f64;

    for i in 0..=n {
        let mid = a + i as f64 * h;
        let l = mid - h;
        let r = mid + h;
        pyramids.push(generate_pyramid(l, r));
    }

    pyramids
}

fn solve(n: usize) {
    let pyramids = create_pyramids(OMEGA_L, OMEGA_R, n);

    // let root_area = BitMapBackend::new("meow.png", (1024, 768)).into_drawing_area();
    // let x_axis = (0.0f64..2.0).step(0.01);
    // root_area.fill(&WHITE);
    // let root_area = root_area.titled("Image Title", ("sans-serif", 60)).unwrap();
    // let (upper, lower) = root_area.split_vertically(512);
    // let mut cc = ChartBuilder::on(&upper)
    //     .margin(5)
    //     .set_all_label_area_size(50)
    //     .caption("Sine and Cosine", ("sans-serif", 40))
    //     .build_cartesian_2d(-3.4f64..3.4, -1.2f64..1.2f64).unwrap();
    // for  pyramid in pyramids.iter() {
    //     cc.draw_series(LineSeries::new(x_axis.values().map(|x| (x, pyramid.d0.f(x))), &RED));
    // }
    // root_area.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");



    let it = pyramids
        .iter()
        .flat_map(|x| pyramids.iter().map(move |y| (y, x)))
        .map(|(w, v)| B(w, v));
    let a_matrix = nalgebra::DMatrix::from_iterator(n + 1, n + 1, it);
    //println!("A matrix for n={}: \n{:#}", n, a_matrix);
    let b_vector = nalgebra::DVector::from_iterator(n + 1, pyramids.iter().map(|v| L(v)));
    let lu: nalgebra::LU<f64, nalgebra::Dyn, nalgebra::Dyn> = a_matrix.clone().lu();
    let result = lu.solve(&b_vector).expect("Failed to solve LU matrix");
    //println!("Result for n={}: {:?}", n, result);
        let root_area = BitMapBackend::new("meow.png", (1024, 768)).into_drawing_area();
    let step = 2.0 / n as f64;
    let x_axis = (0.0f64..2.0).step(step);
    root_area.fill(&WHITE);
    let root_area = root_area.titled("Image Title", ("sans-serif", 60)).unwrap();
    let (upper, lower) = root_area.split_vertically(512);
    let mut cc = ChartBuilder::on(&upper)
        .margin(5)
        .set_all_label_area_size(50)
        .caption("Sine and Cosine", ("sans-serif", 40))
        .build_cartesian_2d(-3.4f64..3.4, 0.0f64..200.0).unwrap();
    cc.draw_series(LineSeries::new(x_axis.values().zip(result.iter()).map(|(x, &beta)| (x, beta + 3.0 * x - 3.0)), &RED));
    root_area.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
}

//hardcoded B(w,v)
#[allow(non_snake_case)]
fn B(w: &C1, v: &C1) -> f64 {
    let e = F { f: Rc::new(E) };
    let left_bound = w.left.max(v.left).max(OMEGA_L);
    let right_bound = w.right.min(v.right).min(OMEGA_R);
    let r = 4.0 * w.d0.f(0.0) * v.d0.f(0.0) - gauss_quadrature(w.d1.clone() * v.d1.clone() * e, left_bound, right_bound);
    return r;
}

#[allow(non_snake_case)]
fn L(v: &C1) -> f64 {
    let e = F { f: Rc::new(E) };
    let sin = F {
        f: Rc::new(|x: f64| f64::sin(PI * x)),
    };
    let const3 = F {
        f: Rc::new(|_: f64| 3.0),
    };
    1000.0 * gauss_quadrature(v.d0.clone() * sin, v.left.max(OMEGA_L), v.right.min(OMEGA_R))
        + gauss_quadrature(const3 * v.d1.clone() * e, v.left.max(OMEGA_L), v.right.min(OMEGA_R))
        + 32.0 * v.d0.f(0.0)
}
