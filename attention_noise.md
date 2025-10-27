A. スペクトル特性に基づく加算ノイズ

A1. ホワイトノイズ（White Gaussian Noise） — A1_white_gaussian
目的：帯域全体に一様な雑音を付与し、基準的なロバスト性を評価する。
概要：周波数特性がフラットなガウス分布ノイズを加算。

A2. ピンクノイズ（Pink Noise, 1/f） — A2_pink_noise_1_over_f
目的：低周波成分の優勢な環境ゆらぎに対する耐性を評価する。
概要：パワースペクトル密度が 1/f に比例するノイズを加算。

A3. ブラウンノイズ（Brown/Red Noise, 1/f²） — A3_brown_noise_1_over_f2
目的：ドリフトや基線うねり等の極低周波ゆらぎへの耐性評価。
概要：パワースペクトル密度が 1/f² に比例するノイズを加算。

A4. ブルーノイズ（Blue Noise, f） — A4_blue_noise_f
目的：高周波成分に偏った干渉への耐性評価。
概要：パワースペクトル密度が f に比例するノイズを加算。

A5. バイオレットノイズ（Violet/Purple Noise, f²） — A5_violet_noise_f2
目的：非常に高周波の微小擾乱に対する感度確認。
概要：パワースペクトル密度が f² に比例するノイズを加算。

A6. 帯域限定ガウスノイズ（Band-limited Gaussian） — A6_band_limited_gaussian
目的：特定周波数帯の干渉（EMI 等）を模擬。
概要：所定のパスバンドのみ通過させたガウスノイズを加算。




B. 構造特性に基づく加算ノイズ

B1. インパルス（スパース）ノイズ — B1_impulse_salt_pepper
目的：突発的スパイクや放電由来の擾乱に対する堅牢性評価。
概要：時刻的に疎な大振幅インパルスを加算。

B2. 電源ハム／高調波（Power-line Hum & Harmonics） — B2_hum_powerline
目的：50/60 Hz 系の周期干渉の影響評価。
概要：基本周波＋高調波の正弦波を重畳。

B3. チャープ干渉（Chirp Interference） — B3_chirp_interference
目的：周波数掃引信号の混入を想定した耐性評価。
概要：時間とともに周波数が変化するトーンを加算。

B4. ランダム・テレグラフ・ノイズ（Random Telegraph Noise） — B4_random_telegraph_noise_rtn
目的：二値レベルのランダム切替に対する挙動評価。
概要：+A/−A の状態が確率的に遷移するノイズを加算。

B5. ショット／ポアソンノイズ（Shot/Poisson Noise） — B5_poisson_shot_noise
目的：離散イベント起源の統計的揺らぎ評価。
概要：ポアソン過程に基づくイベント列を加算。







C. 振幅変調（乗算）型ノイズ

C1. ゲインゆらぎ（Amplitude Modulation） — C1_gain_jitter_am
目的：系全体のゲイン変動への頑健性評価。
概要：信号にランダムな振幅係数を乗算。

C2. エンベロープ・フェージング（Envelope Fading） — C2_envelope_fading
目的：ゆっくり変化する包絡による部分的減衰の影響評価。
概要：低周波包絡を振幅に乗算して局所的に弱化。

C3. 区間ゲインシフト（Block-wise Gain Shift） — C3_blockwise_gain_shift
目的：区間単位の感度変化・接触不良の模擬。
概要：所定区間ごとに定数ゲインを切り替えて乗算。




D. 時間軸擾乱（サンプリング起因）

D1. タイミング・ジッタ（Timing Jitter） — D1_timing_jitter
目的：サンプリング時刻の微小ズレに対する影響評価。
概要：サンプル時刻を確率的に摂動し再サンプリング。

D2. サンプル欠損（Sample Dropouts） — D2_sample_dropouts
目的：記録欠落・通信途絶の模擬。
概要：短区間のサンプル値を欠損（NaN／ゼロ）に置換。

D3. 重複／スキップ（Duplicate/Skip Samples） — D3_duplicate_skip_samples
目的：バッファ異常や読み出し不整合の再現。
概要：同値連続やインデックス飛びを意図的に挿入。

D4. 非線形遅延（Nonlinear Delay） — D4_nonlinear_delay
目的：経路依存の群遅延・フィルタ影響の模擬。
概要：時間軸を位置依存で変形し再配置。





E. 測定器由来の非線形／量子化

E1. DC オフセット — E1_dc_offset
目的：基線シフトに対する補正・頑健性評価。
概要：全体に一定の直流成分を加算。

E2. ベースラインドリフト — E2_baseline_drift
目的：環境起因の緩慢な基線変動の模擬。
概要：低周波成分を加算して基線を時間変化させる。

E3. クリッピング／飽和 — E3_clipping_saturation
目的：計測レンジ超過時の歪み影響評価。
概要：上下限で信号を飽和させる非線形処理。

E4. 多項式歪み — E4_polynomial_distortion
目的：軽度〜中度の非線形応答の模擬。
概要：多項式写像により高調波を付与。

E5. 量子化ビット低下 — E5_quantization_bitdrop
目的：分解能低下・ビット落ちの影響評価。
概要：量子化ビット数を削減して丸め誤差を導入。







F. マスキング／欠落

F1. 時間ブロックアウト — F1_time_blockout
目的：観測不能な連続区間の影響評価。
概要：連続時間ブロックを欠落として扱う。

F2. ランダム時間マスク — F2_random_time_mask
目的：短時間欠落の分散的出現への耐性評価。
概要：短い時間穴をランダムに複数挿入。

F3. 帯域ノッチ（Bandstop/Notch） — F3_bandstop_notch
目的：特定周波数帯の消失・フィルタ欠落の模擬。
概要：指定帯域を抑圧／ゼロ化。








G. 非定常シナリオ

G1. ステップオフセット（Step Offset Jump） — G1_step_offset_jump
目的：スイッチ操作等による突発的基線変化の評価。
概要：時点 t0 でオフセットを段差的に変更。

G2. 周期バースト（Periodic Bursts） — G2_periodic_bursts
目的：周期的擾乱（回転体・ポーリング）の影響評価。
概要：一定周期でノイズバースト区間を挿入。

G3. ランダムウォーク汚染 — G3_random_walk_contamination
目的：累積的な低周波漂移の評価。
概要：ランダムウォーク成分を重畳して長期的変動を生成。

Attention の適用方針（共通記述）

適用位置制御：Attention 値をしきい値化して「適用区間マスク」を作成し、上記各ノイズをその区間に適用する。

強度制御：Attention 値を 0–1 に正規化し、ノイズ強度または適用確率の重みとして用いる（例：係数 = 1 − attention など）。

教師データ化：付与したノイズの種類・区間・強度を必ずログ化し、学習時に「ノイズ区間＝低 Attention／非ノイズ区間＝高 Attention」となるよう損失関数で監督する。