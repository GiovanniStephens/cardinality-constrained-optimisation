"""Tests for the crude name-based ETF classifier (src/categorise.py).

Representative real fund names, focused on the July 2026 hardening (families
that previously fell to Unknown) plus false-positive regression guards for the
substring hazards documented in the module docstring.
"""

import unittest

from src.categorise import classify_etf


class TestNewFixedIncomeFamilies(unittest.TestCase):
    def test_preferred(self):
        self.assertEqual(classify_etf('iShares Preferred and Income Securities ETF'),
                         'Fixed Income')
        self.assertEqual(classify_etf('Invesco Variable Rate Preferred ETF'),
                         'Fixed Income')

    def test_convertible(self):
        self.assertEqual(
            classify_etf('SPDR Bloomberg Barclays Convertible Securities ETF'),
            'Fixed Income')

    def test_clo(self):
        self.assertEqual(classify_etf('Janus Henderson AAA CLO ETF'), 'Fixed Income')

    def test_securitised(self):
        self.assertEqual(classify_etf('iShares CMBS ETF'), 'Fixed Income')
        self.assertEqual(classify_etf('iShares MBS ETF'), 'Fixed Income')
        self.assertEqual(classify_etf('iShares US Mortgage Backed Securities UCITS ETF'),
                         'Fixed Income')

    def test_structured(self):
        self.assertEqual(classify_etf('Obra Opportunistic Structured Products ETF'),
                         'Fixed Income')
        self.assertEqual(
            classify_etf('First Trust Structured Credit Income Opportunities ETF'),
            'Fixed Income')


class TestNewAlternativesFamilies(unittest.TestCase):
    def test_tail_risk(self):
        self.assertEqual(classify_etf('Cambria Tail Risk ETF'), 'Alternatives')

    def test_buywrite(self):
        self.assertEqual(classify_etf('First Trust BuyWrite Income ETF'),
                         'Alternatives')
        self.assertEqual(classify_etf('First Trust Hedged BuyWrite Income ETF'),
                         'Alternatives')

    def test_defined_outcome_family(self):
        self.assertEqual(classify_etf('TrueShares Structured Outcome (October) ETF'),
                         'Alternatives')
        self.assertEqual(classify_etf('Aptus Defined Risk ETF'), 'Alternatives')
        self.assertEqual(classify_etf('Aptus Collared Income Opportunity ETF'),
                         'Alternatives')
        self.assertEqual(classify_etf('Pacer Swan SOS Moderate (July) ETF'),
                         'Alternatives')

    def test_rate_hedge(self):
        self.assertEqual(
            classify_etf('FolioBeyond Alternative Income and Interest Rate Hedge ETF'),
            'Alternatives')

    def test_inflation_expectations(self):
        self.assertEqual(classify_etf('ProShares Inflation Expectations ETF'),
                         'Alternatives')

    def test_rate_hedged_bond_fund_stays_fixed_income(self):
        # 'corporate'/'bond' (Fixed Income) must win over 'interest rate hedge'.
        self.assertEqual(
            classify_etf('iShares Interest Rate Hedged Corporate Bond ETF'),
            'Fixed Income')


class TestNewEquityFamilies(unittest.TestCase):
    def test_sector_thematics(self):
        self.assertEqual(classify_etf('iShares U.S. Aerospace & Defense ETF'), 'Equity')
        self.assertEqual(classify_etf('VanEck Vectors Biotech ETF'), 'Equity')
        self.assertEqual(classify_etf('iShares U.S. Insurance ETF'), 'Equity')
        self.assertEqual(classify_etf('Invesco KBW Regional Banking ETF'), 'Equity')
        self.assertEqual(classify_etf('iShares Transportation Average ETF'), 'Equity')
        self.assertEqual(classify_etf('VanEck Vectors Retail ETF'), 'Equity')
        self.assertEqual(classify_etf('First Trust Cloud Computing ETF'), 'Equity')
        self.assertEqual(classify_etf('Invesco Dynamic Media ETF'), 'Equity')
        self.assertEqual(classify_etf('ALPS Medical Breakthroughs ETF'), 'Equity')
        self.assertEqual(classify_etf('Invesco Solar ETF'), 'Equity')

    def test_brands_and_structures(self):
        self.assertEqual(classify_etf('First Trust United Kingdom AlphaDEX Fund'),
                         'Equity')
        self.assertEqual(classify_etf('First Trust Dorsey Wright Focus 5 ETF'),
                         'Equity')
        self.assertEqual(classify_etf('Invesco QQQ Trust'), 'Equity')
        self.assertEqual(classify_etf('Renaissance IPO ETF'), 'Equity')
        self.assertEqual(classify_etf('AdvisorShares Dorsey Wright ADR ETF'), 'Equity')
        self.assertEqual(classify_etf('Invesco BuyBack Achievers ETF'), 'Equity')
        self.assertEqual(classify_etf('VanEck Vectors Morningstar Wide Moat ETF'),
                         'Equity')
        self.assertEqual(classify_etf('WisdomTree India Earnings Fund'), 'Equity')
        self.assertEqual(classify_etf('Alerian MLP ETF'), 'Equity')
        self.assertEqual(classify_etf('VanEck Vectors BDC Income ETF'), 'Equity')

    def test_resource_equity(self):
        self.assertEqual(classify_etf('VanEck Vectors Natural Resources ETF'), 'Equity')
        self.assertEqual(classify_etf('VanEck Vectors Rare Earth/Strategic Metals ETF'),
                         'Equity')

    def test_single_country(self):
        self.assertEqual(classify_etf('Invesco India ETF'), 'Equity')
        self.assertEqual(classify_etf('VanEck Vectors Vietnam ETF'), 'Equity')
        self.assertEqual(classify_etf('VanEck Vectors Israel ETF'), 'Equity')
        self.assertEqual(classify_etf('JPMorgan BetaBuilders Canada ETF'), 'Equity')
        self.assertEqual(classify_etf('iShares Latin America 40 ETF'), 'Equity')


class TestNewCommodityFamilies(unittest.TestCase):
    def test_base_metals_and_fuel(self):
        self.assertEqual(classify_etf('Invesco DB Base Metals Fund'), 'Commodity')
        self.assertEqual(classify_etf('United States Gasoline Fund, LP'), 'Commodity')


class TestFalsePositiveGuards(unittest.TestCase):
    def test_intermediate_is_not_media(self):
        self.assertEqual(
            classify_etf('Vanguard Intermediate-Term Corporate Bond ETF'),
            'Fixed Income')

    def test_cloud_is_not_clo(self):
        # 'clo ' (trailing space) must not catch cloud/close/clock names.
        self.assertEqual(classify_etf('WisdomTree Cloud Computing Fund'), 'Equity')

    def test_currency_hedged_equity_not_misrouted(self):
        # Prior audit: currency-hedged equity funds must not fall to Currency,
        # and the new terms must not reroute them either.
        self.assertEqual(classify_etf('iShares Currency Hedged MSCI Japan ETF'),
                         'Equity')

    def test_inverse_untouched(self):
        self.assertEqual(classify_etf('ProShares Short Russell2000'), 'Inverse')
        self.assertEqual(classify_etf('Direxion Daily S&P 500 Bear 1X Shares'),
                         'Inverse')

    def test_short_duration_bond_untouched(self):
        self.assertEqual(classify_etf('SPDR Short-Term Corporate Bond ETF'),
                         'Fixed Income')

    def test_alternative_energy_is_equity(self):
        # 'alternative' was deliberately NOT added as an Alternatives term.
        self.assertEqual(classify_etf('iShares Global Alternative Energy ETF'),
                         'Equity')

    def test_dividend_equity_override_untouched(self):
        self.assertEqual(classify_etf('First Trust High Yield Equity Dividend Fund'),
                         'Equity')

    def test_empty_and_none(self):
        self.assertEqual(classify_etf(''), 'Unknown')
        self.assertEqual(classify_etf(None), 'Unknown')

    def test_balanced_allocation_stays_unknown(self):
        # 'allocation' deliberately unclassified — capped, not trusted.
        self.assertEqual(classify_etf('iShares Core Aggressive Allocation ETF'),
                         'Unknown')


if __name__ == '__main__':
    unittest.main()
