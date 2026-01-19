#set text(lang: "pl", size: 12pt)
#set par(justify: true)

= Jakub Własiewicz - Rozwiązanie zadania 4.3 - \ Odkształcenie sprężyste

Równaniem różniczkowym rozwiązanym przez program metodą elementów skończonych jest:
$ - dif / (dif x) (E(x) (dif u)/ (dif x)) = -1000 sin(pi x)\
upright("gdzie") E(x) = cases(2 "dla" x in [0,1], 6 "dla" x in (1,2]) $
Gdzie $[0,2] in.rev x arrow.r.long.bar u(x) in RR$ to szukana funkcja z warunkami brzegowymi:
$ u(2) = 3\ (dif u(0))/(dif x) + 2u(0) = 10 $
 
Wprowadzamy funkcję testową $v$ oraz funkcję $w$ należące do zbioru ${f in H^1(Omega): f(2) = 0}$

Niech $u = w + 3$, jeżeli $u(2) = 3$ to $w(2) = 0$. $w'=u'$\
$u'(0) + 2u(0) = 10 arrow.r.double u'(0) = 10 - 2u(0)$

$ - dif / (dif x) (E(x) (dif u)/ (dif x)) = -1000 sin(pi x) $
$ - (underbrace(E'(x)u', "=0") + E(x)u'') = -1000 sin(pi x) $
$ E(x)u'' = 1000 sin(pi x) $
$ integral_Omega v E(x)u'' d x = 1000 integral_Omega v sin(pi x) d x $
$ lr(u' v E(x)|, size: #200%)^2_0 - integral_0^2 u'(underbrace(E'v, "=0") + v'E) d x = 1000 integral_0^2 v sin (pi x) d x $
$ underbrace(u'(2)v(2)E(2), "=0") - u'(0)v(0)underbrace(E(0), "=2") - integral_0^2 u'v'E d x = 1000 integral_0^2 v sin(pi x) d x $
Wstawiamy warunek lewego brzegu:
$ - 2(10 - 2u(0))v(0) - integral_0^2 u'v'E d x = 1000 integral_0^2 v sin(pi x ) d x $ 
Podstawiamy $w$:
$ -20v(0) + 4v(0)(w(0) + 3) - integral_0^2 w'v'E d x = 1000 integral_0^2 v sin(pi x) d x $
$ 4v(0)w(0) - 8v(0) - integral_0^2 w'v'E d x = 1000 integral_0^2 v sin(pi x) d x $
$ underbrace(4v(0)w(0) - integral_0^2 w'v'E d x, B(w, v)) = underbrace(1000 integral_0^2 v sin(pi x) d x + 8v(0),L(v)) $

Po wyznaczeniu $w$ metodą Galerkina, dodajemy $3$ otrzymując szukaną funkcję $u = w+3$

#figure(
  image("wykres.png"),
  caption: [Wykres rozwiązania]
)

== Instrukcja do uruchomienia programu
Aby uruchomić program należy zainstalować _cargo_. Można to zrobić poniższą komendą: 

`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

Następnie uruchomić `cargo run -r` lub `cargo run` (wersja bez optymalizacji).