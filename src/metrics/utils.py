def calc_cer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        if len(predicted_text) == 0:
            return 0.0
        return 1.0
    return _edit_distance(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    target_words = target_text.split()
    predicted_words = predicted_text.split()
    if len(target_words) == 0:
        if len(predicted_words) == 0:
            return 0.0
        return 1.0
    return _edit_distance(target_words, predicted_words) / len(target_words)


def _edit_distance(a, b):
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]
