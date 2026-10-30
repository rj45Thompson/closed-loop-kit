using System;
using NUnit.Framework;

// Each test says which requirements it checks (/// CHECKS:), written with the test - the loop's L16 holds each requirement's own
// checks to its code: when that code breaks, one of THESE tests must fail.
public class WalletTests
{
    /// CHECKS: W-1
    [Test]
    public void Pay_TakesThePrice_AndOnlyWhatTheWalletHolds()
    {
        var w = new Wallet(100);
        w.Pay(30);
        Assert.That(w.Money, Is.EqualTo(70));
        Assert.That(w.CanPay(70) && !w.CanPay(71), "exactly what it holds, not a coin more");
        Assert.Throws<InvalidOperationException>(() => w.Pay(71));
    }

    /// CHECKS: W-1, W-2
    [Test]
    public void APriceIsNeverNegative_AndFreeIsAPrice()
    {
        var w = new Wallet(0);
        Assert.That(w.CanPay(0), "a free line can be paid from an empty wallet");
        Assert.That(w.CanPay(-1), Is.False, "no negative price");
    }

    /// CHECKS: W-2
    [Test]
    public void AWallet_HoldsAtMostItsDisplay()
    {
        Assert.That(new Wallet(Wallet.Most).Money, Is.EqualTo(Wallet.Most));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Wallet(Wallet.Most + 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Wallet(-1));
        Assert.That(new Wallet(0).Money, Is.EqualTo(0));
    }

    /// CHECKS: W-3
    [Test]
    public void ALimitedLine_SellsItsQuantity_ThenIsSoldOut()
    {
        var s = new Store();
        s.Stock("apple", 10, 2);
        var w = new Wallet(100);
        Assert.That(s.Buy("apple", w), Is.EqualTo(BuyResult.Bought));
        Assert.That(s.Left("apple"), Is.EqualTo(1));
        Assert.That(s.Buy("apple", w), Is.EqualTo(BuyResult.Bought));
        Assert.That(s.Buy("apple", w), Is.EqualTo(BuyResult.SoldOut));
        Assert.That(w.Money, Is.EqualTo(80));
    }

    /// CHECKS: W-3, W-4
    [Test]
    public void AnUnlimitedLine_NeverRunsOut_AndANoMoneyBuyChangesNothing()
    {
        var s = new Store();
        s.Stock("water", 5, -1);
        var w = new Wallet(5);
        Assert.That(s.Buy("water", w), Is.EqualTo(BuyResult.Bought));
        Assert.That(s.Left("water"), Is.EqualTo(-1), "an unlimited line does not count down");
        Assert.That(s.Buy("water", w), Is.EqualTo(BuyResult.NotEnoughMoney));
        Assert.That((w.Money, s.Left("water")), Is.EqualTo((0, -1)));
        Assert.That(s.Left("nothing"), Is.EqualTo(0));
    }
}
