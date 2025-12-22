use std::{clone, f64::consts::PI};
use std::rc::Rc;
use nalgebra;

const OMEGA_L: f64 = 0.0;
const OMEGA_R: f64 = 2.0;

struct C1 {
    pub d0: F, // funkcja
    pub d1: F, // jej pochodna
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
    let f = F { f: Rc::new(|x: f64| x.sin()) };
    println!("{}", gauss_quadrature(f, 0.0, 1.0 * PI));
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
        _ if 0.0 <= x && x <= 1.0 => {2.0},
        _ if 1.0 < x && x <= 2.0 => {6.0},
        _ => {panic!("Left range of function E");}
    }
}

fn generate_pyramid(a: f64, b: f64) -> C1 {
    C1 {
        d0 : F {
            f: Rc::new(move |x: f64| {
                match x {
                    _ if a <= x && x <= (a + b) / 2.0 => {
                        2.0 * (x - a) / (b - a)
                    }
                    _ if (a + b) / 2.0 < x && x <= b => {
                        2.0 * (b - x) / (b - a)
                    }
                    _ => {
                        0.0
                    }
                }
            }),
        },
        d1 : F {
            f: Rc::new(move |x: f64| {
                match x {
                    _ if a <= x && x <= (a + b) / 2.0 => {
                        2.0 / (b - a)
                    }
                    _ if (a + b) / 2.0 < x && x <= b => {
                        -2.0 / (b - a)
                    }
                    _ => {
                        0.0
                    }
                }
            })
        }
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
    let it = pyramids
        .iter()
        .flat_map(|x| {
            pyramids.iter().map(move |y| (y,x))
        })
        .map(|(w, v) | B(w,v));
    let a_matrix = nalgebra::DMatrix::from_iterator(n + 1, n + 1, it);
    let b_vector = nalgebra::DVector::from_iterator(n + 1, pyramids.iter().map(|v| L(v)));

}


//hardcoded B(w,v)
#[allow(non_snake_case)]
fn B(w: &C1, v: &C1) -> f64 {
    let e = F { f: Rc::new(E) };
    4.0 * w.d0.f(0.0) * v.d0.f(0.0) - gauss_quadrature(w.d1.clone() * v.d1.clone() * e, OMEGA_L, OMEGA_R)
}

#[allow(non_snake_case)]
fn L(v: &C1) -> f64 {
    let e = F { f: Rc::new(E) };
    let sin = F { f: Rc::new(|x: f64| f64::sin(PI * x)) };
    let const3 = F { f: Rc::new(|_: f64| 3.0) };
    1000.0 * gauss_quadrature(v.d0.clone() * sin, OMEGA_L, OMEGA_R) + gauss_quadrature(const3 * v.d1.clone() * e, OMEGA_L, OMEGA_R) + 32.0 * v.d0.f(0.0)
}
