import numpy as np
import math
import matplotlib.pyplot as plt

class SDAR:
    """Sequentially Discounting Autoregressive (SDAR) model.

    ARモデルのパラメータ（平均、自己共分散・AR係数、予測誤差分散）をオンラインで更新する。

    Attributes:
        order: ARモデルの次数
        r: 忘却パラメータ (0 < r < 1)
        mean: μ の値
        var: σ^2 の値
        C: 自己共分散推定値 C_j (j=0,...,order)
        ar_coef: AR係数 a_1,...,a_order
        buffer: 過去order個の観測値

    Note:
        更新式:
            μ ← (1 - r) μ + r x_t
            C_j ← (1 - r) C_j + r (x_t - μ)(x_{t-j} - μ)   (j=0,...,order)
            Yule-Walker方程式: solve [C(|i-j|)]_{i,j=0}^{order-1} * a = [C_1, ..., C_order]^T
            予測: x̂_t = μ + sum_{i=1}^{order} a_i (x_{t-i} - μ)
            分散: σ^2 ← (1 - r) σ^2 + r (x_t - x̂_t)^2
            スコア: -log p(x_t) = 0.5*log(2πσ^2) + 0.5*( (x_t - x̂_t)^2 / σ^2 )
    """

    def __init__(self, order, r, init_mean=0.0, init_var=1.0):
        """SDARモデルの初期化を行う。

        Args:
            order: ARモデルの次数
            r: 忘却パラメータ (0 < r < 1)
            init_mean: μの初期値 (default: 0.0)
            init_var: σ^2の初期値 (default: 1.0)
        """
        self.order = order
        self.r = r
        self.mean = init_mean
        self.var = init_var
        self.C = np.zeros(order + 1)
        self.ar_coef = np.zeros(order)
        self.buffer = []

    def update(self, x):
        """新たな観測値でARモデルのパラメータを更新する。

        Args:
            x: 新しい観測値 (float)

        Returns:
            tuple:
                - x_pred (float or None): ARモデルによる予測値
                - score (float or None): 外れ値スコア
        """
        self.mean = (1 - self.r) * self.mean + self.r * x

        if len(self.buffer) < self.order:
            self.buffer.append(x)
            diff = x - self.mean
            self.C[0] = (1 - self.r) * self.C[0] + self.r * (diff * diff)
            return None, None

        diff = x - self.mean
        self.C[0] = (1 - self.r) * self.C[0] + self.r * (diff * diff)

        for j in range(1, self.order + 1):
            x_lag = self.buffer[-j]
            diff_lag = x_lag - self.mean
            self.C[j] = (1 - self.r) * self.C[j] + self.r * (diff * diff_lag)

        T = np.empty((self.order, self.order))
        for i in range(self.order):
            for j in range(self.order):
                T[i, j] = self.C[abs(i - j)]
        rhs = self.C[1:self.order + 1]

        try:
            self.ar_coef = np.linalg.solve(T, rhs)
        except np.linalg.LinAlgError:
            self.ar_coef = np.zeros(self.order)

        x_pred = self.mean
        for i in range(self.order):
            x_pred += self.ar_coef[i] * (self.buffer[-1 - i] - self.mean)

        error = x - x_pred
        self.var = (1 - self.r) * self.var + self.r * (error ** 2)
        score = 0.5 * math.log(2 * math.pi * self.var) + 0.5 * ((error ** 2) / self.var)

        self.buffer.append(x)
        if len(self.buffer) > self.order:
            self.buffer.pop(0)

        return x_pred, score

class ChangeFinder:
    """時系列データの変化点を検出するChangeFinderアルゴリズム。

    2段階のSDARモデルを用いて変化点を検出する。

    Attributes:
        stage1: 1段階目のSDARモデル
        stage2: 2段階目のSDARモデル
        smooth_window1: 1段階目の平滑化窓幅
        smooth_window2: 2段階目の平滑化窓幅
        score_buffer1: 1段階目のスコアバッファ
        score_buffer2: 2段階目のスコアバッファ
        change_scores: 各時刻の変化点スコア
    """

    def __init__(self, order1=5, order2=5, r=0.1, smooth_window1=25, smooth_window2=25):
        """ChangeFinderの初期化を行う。

        Args:
            order1: 1段階目のARモデル次数 (default: 5)
            order2: 2段階目のARモデル次数 (default: 5)
            r: 忘却パラメータ (default: 0.1)
            smooth_window1: 1段階目の平滑化窓幅 (default: 25)
            smooth_window2: 2段階目の平滑化窓幅 (default: 25)
        """
        self.stage1 = SDAR(order=order1, r=r)
        self.stage2 = SDAR(order=order2, r=r)
        self.smooth_window1 = smooth_window1
        self.smooth_window2 = smooth_window2
        self.score_buffer1 = []
        self.score_buffer2 = []
        self.change_scores = []
    
    def update(self, x):
        """
        新たな観測 x に対して ChangeFinder の処理を行い，変化点スコアを返す．
        
        処理の流れ：
          1. stage1 により x の外れ値スコア score1 を算出
          2. score1 の最近 smooth_window1 個の平均を y としてスムージング
          3. stage2 により y に対するスコア score2 を算出
          4. score2 の最近 smooth_window2 個の平均を最終の変化点スコアとする
        """
        # 【Step 1】 第１段階：生系列 x に対して SDAR を適用
        _, score1 = self.stage1.update(x)
        if score1 is None:
            # 十分な履歴がなければ 0 を仮の値とする
            score1 = 0.0
        self.score_buffer1.append(score1)
        # 窓幅 smooth_window1 でスムージング
        if len(self.score_buffer1) >= self.smooth_window1:
            y = np.mean(self.score_buffer1[-self.smooth_window1:])
        else:
            y = np.mean(self.score_buffer1)
        
        # 【Step 3】 第２段階：スムージング後の y に対して SDAR を適用
        _, score2 = self.stage2.update(y)
        if score2 is None:
            score2 = 0.0
        self.score_buffer2.append(score2)
        # 窓幅 smooth_window2 でスムージングして最終変化点スコアとする
        if len(self.score_buffer2) >= self.smooth_window2:
            final_score = np.mean(self.score_buffer2[-self.smooth_window2:])
        else:
            final_score = np.mean(self.score_buffer2)
        
        self.change_scores.append(final_score)
        return final_score

    def process(self, data):
        """
        与えられた時系列データ（リストまたは 1次元 ndarray）に対して，
        時系列全体の変化点スコアを計算し，リストとして返す．
        """
        scores = []
        for x in data:
            score = self.update(x)
            scores.append(score)
        return scores
