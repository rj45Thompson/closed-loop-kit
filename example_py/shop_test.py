"""The Python example's tests: each prints `selftest: <what it checks> PASS|FAIL` and the script exits 1 on any failure - the form
the loop's Python adapter reads. Each test names the requirement it checks first (CHECKS: W-n).

Each check builds what it needs and runs inside its own try: a break that makes the code RAISE fails that check by name, instead of
ending the script (10-29: once the loop read Python check lines, a crash was credited to "the script" and to no requirement's own
check, so the example held only while L16 was blind to Python)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shop import MOST, Store, Wallet  # noqa: E402

results = []


def check(name, fn):
    try:
        ok = bool(fn())
    except Exception:  # noqa: BLE001 - a raise is this check's failure, reported under its own name
        ok = False
    results.append(ok)
    print("selftest: %s %s" % (name, "PASS" if ok else "FAIL"))


def raises(f, *a):
    try:
        f(*a)
    except ValueError:
        return True
    return False


def paid(money, price):
    w = Wallet(money)
    w.pay(price)
    return w


def store(*lines):
    s = Store()
    for item, price, qty in lines:
        s.stock(item, price, qty)
    return s


def buys(s, item, w, n):
    return [s.buy(item, w) for _ in range(n)]


# CHECKS: W-1 - the wallet pays only what it holds, never a negative price
check("W-1 paying exactly what it holds empties it", lambda: paid(100, 100).money == 0)
check("W-1 a price above the money is refused", lambda: (lambda w: raises(w.pay, 101) and w.money == 100)(Wallet(100)))
check("W-1 a negative price is refused", lambda: (lambda w: raises(w.pay, -1) and w.money == 100)(Wallet(100)))
check("W-1 can_pay: yes to what it holds, no to one more, no to a negative",
      lambda: (lambda w: w.can_pay(100) is True and w.can_pay(101) is False and w.can_pay(-1) is False and w.can_pay(0) is True)(Wallet(100)))
check("W-1 paying takes the price off", lambda: paid(100, 30).money == 70)

# CHECKS: W-2 - a wallet holds from nothing up to what its eight-digit display shows
check("W-2 nothing and the most are allowed", lambda: Wallet(0).money == 0 and Wallet(MOST).money == MOST and MOST == 99999999)
check("W-2 below nothing and above the most are refused", lambda: raises(Wallet, -1) and raises(Wallet, MOST + 1))

# CHECKS: W-3 - a limited line sells its quantity, then is sold out; an unlimited line never runs out
check("W-3 a limited line counts down",
      lambda: (lambda s, w: s.buy("tea", w) == "bought" and s.left("tea") == 1 and w.money == 970)(store(("tea", 30, 2)), Wallet(1000)))
check("W-3 the last one sells, then it is sold out",
      lambda: (lambda s, w: buys(s, "tea", w, 3) == ["bought", "bought", "sold out"] and s.left("tea") == 0 and w.money == 940)(
          store(("tea", 30, 2)), Wallet(1000)))
check("W-3 a line never stocked has none left and is sold out",
      lambda: (lambda s, w: s.left("cake") == 0 and s.buy("cake", w) == "sold out")(store(), Wallet(1000)))

# CHECKS: W-3, W-4 - an unlimited line never runs out, and a purchase refused for money changes neither the wallet nor the stock
# (the C# example's AnUnlimitedLine_NeverRunsOut_AndANoMoneyBuyChangesNothing: both rules live in Store.buy, so both name these)
check("W-3 an unlimited line never runs out",
      lambda: (lambda s, w: buys(s, "water", w, 3) == ["bought"] * 3 and s.left("water") == -1 and w.money == 985)(
          store(("water", 5, -1)), Wallet(1000)))
check("W-4 a purchase it cannot pay is refused and nothing changes",
      lambda: (lambda s, w: s.buy("tea", w) == "not enough money" and w.money == 29 and s.left("tea") == 2)(store(("tea", 30, 2)), Wallet(29)))

sys.exit(0 if all(results) else 1)
