using System;
using System.Collections.Generic;

// The closed loop kit's example: a store's wallet and its stock, small enough to read in a minute and real enough that every
// requirement in example/model has code a test must catch breaking. The installer runs the whole loop on it as its acceptance test.

/// <summary>A player's money: it pays only what it holds, never a negative price, and never more than its display shows.</summary>
public sealed class Wallet
{
    /// <summary>The most the wallet's display can show (eight digits).</summary>
    public const int Most = 99999999;
    public int Money { get; private set; }

    public Wallet(int money)
    {
        if (money < 0 || money > Most) throw new ArgumentOutOfRangeException(nameof(money));
        Money = money;
    }

    public bool CanPay(int price) => price >= 0 && Money >= price;

    public void Pay(int price)
    {
        if (!CanPay(price)) throw new InvalidOperationException("cannot pay " + price);
        Money -= price;
    }
}

/// <summary>What buying came to.</summary>
public enum BuyResult { Bought, NotEnoughMoney, SoldOut }

/// <summary>A store's stock: a line is unlimited, or has a quantity that sells out.</summary>
public sealed class Store
{
    readonly Dictionary<string, int> left = new Dictionary<string, int>(StringComparer.Ordinal);
    readonly Dictionary<string, int> price = new Dictionary<string, int>(StringComparer.Ordinal);

    /// <summary>Adds a line; quantity -1 never runs out.</summary>
    public void Stock(string item, int itemPrice, int quantity)
    {
        price[item] = itemPrice;
        left[item] = quantity;
    }

    public int Left(string item) => left.TryGetValue(item, out int n) ? n : 0;

    /// <summary>One purchase: sold out first, then the money; a limited line counts down by one.</summary>
    public BuyResult Buy(string item, Wallet wallet)
    {
        if (Left(item) == 0) return BuyResult.SoldOut;
        if (!wallet.CanPay(price[item])) return BuyResult.NotEnoughMoney;
        wallet.Pay(price[item]);
        if (left[item] > 0) left[item] -= 1;
        return BuyResult.Bought;
    }
}
