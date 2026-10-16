"""Settings and name rules for prepare_dataset.py."""

SNAPSHOT = "2026-10-01"
SOURCE_URL = f"https://download.companieshouse.gov.uk/BasicCompanyDataAsOneFile-{SNAPSHOT}.zip"
SOURCE_SHA256 = "29d54c17e5bf39862a08d5ebf1ae83abc079c8591c8c6d1bab0a29879accb4ee"
SEED = 42
SAMPLE_SIZE = 100_000
CATEGORY = "Private Limited Company"
MAX_LENGTH = 60
# capital letters, digits, the space and the punctuation that appears often in the register
ALLOWED_CHARACTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 &.()-',/+@!")
NAME_ENDINGS = ("LIMITED", "LTD", "LTD.")
# shares of the name groups for test and validation, the rest is for training
TEST_SHARE = 0.10
VALIDATION_SHARE = 0.10

# key for the register check, following sections 6.4 and 6.5 of "Incorporation and names",
# https://www.gov.uk/government/publications/incorporation-and-names/incorporation-and-names
# letters that Unicode does not split into a base letter and an accent
SPECIAL_LETTERS = {"Æ": "AE", "Ǽ": "AE", "Œ": "OE", "Ø": "O", "Ǿ": "O", "Ł": "L", "Ŀ": "L",
                   "Þ": "D", "Đ": "D", "Ħ": "H", "Ŧ": "T", "Ŋ": "N"}
SYMBOLS = {"@": " AT ", "+": " PLUS ", "%": " PERCENTUM ", "£": " POUND ", "€": " EURO ",
           "$": " DOLLAR ", "¥": " YEN ", "&": " AND "}
WORDS = {"TWO": "TOO", "TO": "TOO", "ZERO": "O", "FOUR": "FOR", "PERCENT": "PERCENTUM"}
DIGITS = {"0": "O", "1": "ONE", "2": "TOO", "3": "THREE", "4": "FOR", "5": "FIVE", "6": "SIX",
          "7": "SEVEN", "8": "EIGHT", "9": "NINE"}
# legal forms in English and Welsh that are ignored at the end of a name, without punctuation
LEGAL_ENDINGS = [
    "LIMITED", "UNLIMITED", "PUBLIC LIMITED COMPANY", "COMMUNITY INTEREST COMPANY",
    "COMMUNITY INTEREST PUBLIC LIMITED COMPANY", "RIGHT TO ENFRANCHISEMENT", "RIGHT TO MANAGE",
    "UK EEIG", "INVESTMENT COMPANY WITH VARIABLE CAPITAL", "LIMITED PARTNERSHIP",
    "LIMITED LIABILITY PARTNERSHIP", "OPEN ENDED INVESTMENT COMPANY",
    "CHARITABLE INCORPORATED ORGANISATION", "INDUSTRIAL AND PROVIDENT SOCIETY",
    "CO OPERATIVE SOCIETY", "COMMUNITY BENEFIT SOCIETY", "CYFYNGEDIG", "ANGHYFYNGEDIG",
    "CWMNI CYFYNGEDIG CYHOEDDUS", "CWMNI BUDDIANT CYMUNEDOL",
    "CWMNI BUDDIANT CYMUNEDOL CYHOEDDUS CYFYNGEDIG", "HAWL I RYDDFREINIAD", "CWMNI RTM CYFYNGEDIG",
    "CWMNI BUDDSODDI A CHYFALAF NEWIDIOL", "PARTNERIAETH CYFYNGEDIG",
    "PARTNERIAETH ATEBOLRWYDD CYFYNGEDIG", "CWMNI BUDDSODDIAD PENAGORED",
    "SEFYDLIAD ELUSENNOL CORFFOREDIG", "LTD", "PLC", "CIC", "RTE", "RTM", "LP", "LLP", "CIO", "CYF",
    "CCC", "CBC", "COMMUNITY INTEREST PLC", "CWMNI BUDDIANT CYMUNEDOL CCC", "PC", "PAC", "SEC",
]
# expressions that are ignored at the end of a name after a space or a full stop
OTHER_ENDINGS = [
    "AND CO", "AND COMPANY", "BIZ", "CO", "CO UK", "COM", "COMPANY", "EU", "GB", "GREAT BRITAIN",
    "NET", "NI", "NORTHERN IRELAND", "ORG", "ORG UK", "UK", "UNITED KINGDOM", "WALES", "AND CWMNI",
    "A R CWMNI", "CWMNI", "CYM", "CYMRU", "DU", "PF", "PRYDAIN FAWR", "Y DEYRNAS UNEDIG",
]

# words that are ignored at the end of a name for the name groups of the split, a shorter list
# than above, because the groups only have to catch near copies and not follow the official rules
IGNORED_ENDINGS = ["LIMITED", "LTD", "PLC", "LLP", "CYFYNGEDIG", "CYF", "UNLIMITED",
                   "AND CO", "AND COMPANY", "CO", "COM", "NET", "ORG", "BIZ", "EU",
                   "UK", "UNITED KINGDOM", "GB", "GREAT BRITAIN", "NI", "NORTHERN IRELAND", "WALES"]
