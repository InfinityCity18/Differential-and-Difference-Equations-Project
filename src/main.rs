use nalgebra;
use std::rc::Rc;
use std::f64::consts::PI;
use plotters::prelude::*;

const OMEGA_L: f64 = 0.0;
const OMEGA_R: f64 = 2.0;

struct C1 {
    pub d0: F, // funkcja
    pub d1: F, // jej pochodna
    pub left: f64, // lewy kraniec przedzialu
    pub right: f64, // prawy kraniec przedzialu
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut n = String::new();
    let mut filename= String::new();
    println!("Podaj n:");
    std::io::stdin().read_line(&mut n)?;
    println!("Podaj nazwę pliku do zapisu wykresu:");
    std::io::stdin().read_line(&mut filename)?;
    filename = filename.trim().to_owned() + ".png";

    let n= n.trim().parse::<usize>()?;
    println!("Rozwiązywanie w toku.");
    let res = solve(n);
    println!("Rozwiązano równanie. Rysowanie w toku.");
    plot(OMEGA_L, OMEGA_R, n, res, &filename)?;
    println!("Wykres gotowy.");

    return Ok(());
}

fn gauss_quadrature(f: F, a: f64, b: f64) -> f64 {
    let inv_sqrt_3: f64 = 1.0 / 3f64.sqrt();
    let b_m_a = (b - a) / 2.0;
    let a_p_b = (a + b) / 2.0;

    b_m_a * (f.f(b_m_a * inv_sqrt_3 + a_p_b) + f.f(b_m_a * -inv_sqrt_3 + a_p_b))
}

fn gauss_quadrature2(f: F, a: f64, b: f64) -> f64 {
    let b_m_a = (b - a) / 2.0;
    let a_p_b = (a + b) / 2.0;

    let mut sum = 0.0;
    for (weight, point) in GAUSS_POINTS.iter() {
        sum += weight * f.f(b_m_a * point + a_p_b);
    }
    return b_m_a * sum;
}

#[allow(non_snake_case)]
fn E(x: f64) -> f64 {
    if 0.0 <= x && x <= 1.0 {
        2.0
    } else if 1.0 < x && x <= 2.0 {
        6.0
    } else {
        panic!("Left range of function E");
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

fn plot(a: f64, b: f64, n: usize, points: Vec<f64>, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let y_max = points.iter().max_by(|a,b| a.total_cmp(b)).ok_or("no max value")?;
    let y_min = points.iter().min_by(|a,b| a.total_cmp(b)).ok_or("no min value")?;
    let diff = (y_max - y_min).abs() * 0.2;

    let root = BitMapBackend::new(filename, (1920,1080)).into_drawing_area();
    let step = (b - a) / n as f64;
    let x_axis = (a..b+step).step(step);
    root.fill(&WHITE)?;
    let root = root.titled("Differential equation solution", ("sans-serif", 60))?;

    let mut cc = ChartBuilder::on(&root)
        .margin(5)
        .set_all_label_area_size(50)
        .caption("u(x)", ("sans-serif", 40))
        .build_cartesian_2d(a..b, y_min-diff..y_max+diff).unwrap();
    cc.configure_mesh().draw()?;
    cc.draw_series(LineSeries::new(x_axis.values().zip(points.iter().cloned()), &RED))?;

    return Ok(());
}

fn solve(mut n: usize) -> Vec<f64> {
    let mut pyramids = create_pyramids(OMEGA_L, OMEGA_R, n);
    pyramids.pop();
    n -= 1;
    let it = pyramids
        .iter()
        .flat_map(|x| pyramids.iter().map(move |y| (y, x)))
        .map(|(w, v)| B(w, v));
    let a_matrix = nalgebra::DMatrix::from_iterator(n + 1, n + 1, it);
    let b_vector = nalgebra::DVector::from_iterator(n + 1, pyramids.iter().map(|v| L(v)));
    let lu: nalgebra::LU<f64, nalgebra::Dyn, nalgebra::Dyn> = a_matrix.clone().lu();
    let result = lu.solve(&b_vector).expect("Failed to solve LU matrix");
    return result.iter().cloned().chain(std::iter::once(0.0)).map(|x| x + 3.0).collect();
}

//hardcoded B(w,v)
// #[allow(non_snake_case)]
// fn B(w: &C1, v: &C1) -> f64 {
//     let e = F { f: Rc::new(E) };
//     let left_bound = w.left.max(v.left).max(OMEGA_L);
//     let right_bound = w.right.min(v.right).min(OMEGA_R);
//     let r = 4.0 * w.d0.f(0.0) * v.d0.f(0.0) - gauss_quadrature(w.d1.clone() * v.d1.clone() * e, left_bound, right_bound);
//     return r;
// }

// #[allow(non_snake_case)]
// fn L(v: &C1) -> f64 {
//     let e = F { f: Rc::new(E) };
//     let sin = F {
//         f: Rc::new(|x: f64| f64::sin(PI * x)),
//     };
//     let const3 = F {
//         f: Rc::new(|_: f64| 3.0),
//     };
//     1000.0 * gauss_quadrature(v.d0.clone() * sin, v.left.max(OMEGA_L), v.right.min(OMEGA_R))
//         + gauss_quadrature(const3 * v.d1.clone() * e, v.left.max(OMEGA_L), v.right.min(OMEGA_R))
//         + 32.0 * v.d0.f(0.0)
// }

#[allow(non_snake_case)]
fn L(v: &C1) -> f64 {
    let sin = F {
        f: Rc::new(|x: f64| f64::sin(PI * x)),
    };
    1000.0 * gauss_quadrature2(v.d0.clone() * sin, v.left.max(OMEGA_L), v.right.min(OMEGA_R)) + 8.0 * v.d0.f(0.0)
}

#[allow(non_snake_case)]
fn B(w: &C1, v: &C1) -> f64 {
    let e = F { f: Rc::new(E) };
    let left_bound = w.left.max(v.left).max(OMEGA_L);
    let right_bound = w.right.min(v.right).min(OMEGA_R);
    if left_bound <= 1.0 && 1.0 < right_bound {
        let r1 = 4.0 * w.d0.f(0.0) * v.d0.f(0.0)
            - gauss_quadrature2(
                w.d1.clone() * v.d1.clone() * e.clone(),
                left_bound,
                1.0,
            )
            - gauss_quadrature2(
                w.d1.clone() * v.d1.clone() * e.clone(),
                1.0,
                right_bound,
            );
        r1
    } else {
        4.0 * w.d0.f(0.0) * v.d0.f(0.0) - gauss_quadrature2(w.d1.clone() * v.d1.clone() * e, left_bound, right_bound)
    }
}

const GAUSS_POINTS: [(f64, f64); 64] = [
(0.0486909570091397,-0.0243502926634244),
(0.0486909570091397,0.0243502926634244 ),
(0.0485754674415034,-0.072993121787799 ),
(0.0485754674415034,0.072993121787799  ),
(0.048344762234803,-0.121462819296121  ),
(0.048344762234803,0.121462819296121   ),
(0.0479993885964583,-0.169644420423993 ),
(0.0479993885964583,0.169644420423993  ),
(0.0475401657148303,-0.217423643740007 ),
(0.0475401657148303,0.217423643740007  ),
(0.04696818281621,-0.264687162208767   ),
(0.04696818281621,0.264687162208767    ),
(0.0462847965813144,-0.311322871990211 ),
(0.0462847965813144,0.311322871990211  ),
(0.0454916279274181,-0.357220158337668 ),
(0.0454916279274181,0.357220158337668  ),
(0.0445905581637566,-0.402270157963992 ),
(0.0445905581637566,0.402270157963992  ),
(0.0435837245293235,-0.446366017253464 ),
(0.0435837245293235,0.446366017253464  ),
(0.0424735151236536,-0.489403145707053 ),
(0.0424735151236536,0.489403145707053  ),
(0.0412625632426235,-0.531279464019895 ),
(0.0412625632426235,0.531279464019895  ),
(0.0399537411327203,-0.571895646202634 ),
(0.0399537411327203,0.571895646202634  ),
(0.0385501531786156,-0.611155355172393 ),
(0.0385501531786156,0.611155355172393  ),
(0.03705512854024,-0.648965471254657   ),
(0.03705512854024, 0.648965471254657    ),
(0.0354722132568824,-0.685236313054233 ),
(0.0354722132568824,0.685236313054233  ),
(0.0338051618371416,-0.719881850171611 ),
(0.0338051618371416,0.719881850171611  ),
(0.0320579283548516,-0.752819907260532 ),
(0.0320579283548516,0.752819907260532  ),
(0.0302346570724025,-0.783972358943341 ),
(0.0302346570724025,0.783972358943341  ),
(0.0283396726142595,-0.813265315122798 ),
(0.0283396726142595,0.813265315122798  ),
(0.0263774697150547,-0.84062929625258  ),
(0.0263774697150547,0.84062929625258   ),
(0.0243527025687109,-0.865999398154093 ),
(0.0243527025687109,0.865999398154093  ),
(0.0222701738083833,-0.889315445995114 ),
(0.0222701738083833,0.889315445995114  ),
(0.0201348231535302,-0.910522137078503 ),
(0.0201348231535302,0.910522137078503  ),
(0.0179517157756973,-0.92956917213194  ),
(0.0179517157756973,0.92956917213194   ),
(0.0157260304760247,-0.946411374858403 ),
(0.0157260304760247,0.946411374858403  ),
(0.0134630478967186,-0.961008799652054 ),
(0.0134630478967186,0.961008799652054  ),
(0.0111681394601311,-0.973326827789911 ),
(0.0111681394601311,0.973326827789911  ),
(0.0088467598263639,-0.983336253884626 ),
(0.0088467598263639,0.983336253884626  ),
(0.0065044579689784,-0.991013371476744 ),
(0.0065044579689784,0.991013371476744  ),
(0.0041470332605625,-0.996340116771955 ),
(0.0041470332605625,0.996340116771955  ),
(0.0017832807216964,-0.999305041735772 ),
(0.0017832807216964,0.999305041735772  )];