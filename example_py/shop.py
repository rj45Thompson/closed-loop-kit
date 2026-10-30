"""The kit's Python example: a wallet and a store, the same rules as the C# example (example/). The installer runs the whole loop
on it when there is no .NET SDK, so every machine with Python can run an acceptance test."""

MOST = 99999999          # the most an eight-digit display shows


class Wallet(object):
    """A player's money: never below nothing, never more than the display can show."""

    def __init__(self, money):
        if money < 0 or money > MOST:
            raise ValueError("a wallet holds 0 to %d" % MOST)
        self.money = money

    def can_pay(self, price):
        return price >= 0 and self.money >= price

    def pay(self, price):
        if not self.can_pay(price):
            raise ValueError("cannot pay %d from %d" % (price, self.money))
        self.money -= price


class Store(object):
    """Lines on sale, each with a price and a quantity; -1 never runs out."""

    def __init__(self):
        self.prices, self.stock_left = {}, {}

    def stock(self, item, price, quantity):
        self.prices[item] = price
        self.stock_left[item] = quantity

    def left(self, item):
        return self.stock_left.get(item, 0)

    def buy(self, item, wallet):
        n = self.left(item)
        if n == 0:
            return "sold out"
        if not wallet.can_pay(self.prices[item]):
            return "not enough money"
        wallet.pay(self.prices[item])
        if n > 0:
            self.stock_left[item] = n - 1
        return "bought"
