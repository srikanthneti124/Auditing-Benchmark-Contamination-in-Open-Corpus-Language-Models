from overlap import code_windows, windows


def words(n):
    return " ".join(f"w{i}" for i in range(n))


def test_empty_text():
    assert windows("", 13, 15, 10) == []
    assert code_windows("", 32, 15, 16) == []


def test_shorter_than_min_words_gives_nothing():
    assert windows(words(9), 13, 15, 10) == []


def test_between_min_words_and_n_gives_whole_text():
    assert windows(words(10), 13, 15, 10) == [words(10)]
    assert windows(words(12), 13, 15, 10) == [words(12)]


def test_exactly_n_words_gives_one_window():
    assert windows(words(13), 13, 15, 10) == [words(13)]


def test_all_windows_when_fewer_than_k():
    ws = windows(words(20), 13, 15, 10)
    assert len(ws) == 8
    assert all(len(w.split()) == 13 for w in ws)


def test_subsampling_keeps_first_and_last_window():
    ws = windows(words(100), 13, 15, 10)
    assert len(ws) == 15
    assert ws[0] == " ".join(f"w{i}" for i in range(13))
    assert ws[-1] == " ".join(f"w{i}" for i in range(87, 100))


def test_k_equal_one_does_not_crash():
    assert windows(words(100), 13, 1, 10) == [" ".join(f"w{i}" for i in range(13))]


def test_whitespace_is_normalised_for_text():
    assert windows("a  b\n\tc " * 5, 3, 15, 1)[0] == "a b c"


def test_code_windows_preserve_original_whitespace():
    src = "def f(x):\n    return x + 1\n"
    ws = code_windows(src, 4, 15, 1)
    assert ws[0] == "def f(x):\n    return x"
    assert ws[-1] == "return x + 1"
    assert all(w in src for w in ws)


def test_code_windows_short_text():
    assert code_windows("def f(x):\n    pass\n", 32, 15, 16) == []
    assert code_windows("  def f(x):\n    pass\n", 32, 15, 3) == ["def f(x):\n    pass"]
