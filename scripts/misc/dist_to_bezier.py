import sympy
import IPython


def main():
    t, p, v1, v2, v3 = sympy.symbols("t, p, v1, v2, v3")
    b = (1 - t)**2 * v1 + 2*(1 - t) * t * v2 + t**2 * v3
    J = (p - b)**2
    dJdt = J.diff(t).expand()
    print(f"a = {dJdt.coeff(t, 3)}")
    print(f"b = {dJdt.coeff(t, 2)}")
    print(f"c = {dJdt.coeff(t, 1)}")
    print(f"d = {dJdt.coeff(t, 0)}")

    IPython.embed()


if __name__ == "__main__":
    main()
