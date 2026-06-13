from spec_bpe.utils import DataProcessor

def test_code_mode():
    dp = DataProcessor(code_mode=True)
    code = "def  func():\n    return  True"
    processed = dp.process_corpus(code)
    assert "def func():" in processed
    assert "\n" in processed
    assert "return True" in processed

def test_standard_mode():
    dp = DataProcessor(preserve_newlines=False, code_mode=False)
    text = "Hello   \n world"
    processed = dp.process_corpus(text)
    assert processed == "Hello world"
