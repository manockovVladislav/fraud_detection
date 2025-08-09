# Multi-Task Fraud Detection (MTL + Cost-Sensitive)

Совместная модель для детекции мошенничества:
- **Классификация**: вероятность мошенничества (hat{y}).
- **Регрессия**: оценка ущерба \(\hat{r}\).
- **Функция стоимости**: оптимизация с учётом цены ошибок (FN/FP).
- **Бейзлайн**: XGBoost с `scale_pos_weight`.
- **Метрики**: ROC-AUC, PR-кривые, отчёты по классам.
- **Риск-скоринг**: \(\hat{s}=\hat{y}\cdot\hat{r}\) для приоритизации расследований.

## 1) Идея и мотивация

Финансовые потери из-за FN превышают издержки на FP. Обычные классификаторы оптимизируют «усреднённые» метрики и не различают «дорогие» ошибки.  
Многозадачное обучение (MTL) + стоимостно-чувствительная классификация дают лучшее извлечение общих представлений, чувствительность к крупным потерям и бизнес-релевантную оптимизацию.

## 2) Архитектура

Общий блок и две «головы»:

```
Input (d) → Linear(128) → ReLU → Linear(64) → ReLU
                      ├─→ Head_cls: Linear(1) → Sigmoid → ŷ
                      └─→ Head_reg: Linear(1)          → r̂
```

Математика:
- Выходы: \(\hat{y}_i=\sigma(W_c h_\theta(x_i)+b_c),\;\hat{r}_i=W_r h_\theta(x_i)+b_r\).
- Итоговый лосс: \(L_{\text{total}}=\alpha L_{\text{cls}}+\beta L_{\text{reg}}\).
- Классификация (focal): \(L_{\text{focal}}=-\tfrac{1}{N}\sum_i \alpha_f (1-p_{t,i})^\gamma \log p_{t,i}\).
- Регрессия (MSE по y=1): \(L_{\text{reg}}=\tfrac{1}{N_1}\sum_{i:y_i=1}(\hat{r}_i-r_i)^2\).
- Риск-скоринг: \(\hat{s}_i=\hat{y}_i\hat{r}_i\).

## 3) Датасет

- **Credit Card Fraud** (ULB). Дисбаланс ≈ 0.172% мошенничеств.  
- Предобработка: `StandardScaler` для `Amount`, `Time`.  
- Положи CSV в `data/creditcard.csv` или укажи путь через `--data_path`.

## 4) Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 5) Гиперпараметры и cost-sensitive настройка

- `alpha_reg=0.1`, `gamma=2`, `alpha_f=0.25`  
- `scale_pos_weight ≈ N_neg/N_pos` для XGBoost  
- Порог \(\tau\) подбираем по минимуму:  
  \(\mathrm{Cost}(\tau)=\sum_{FN}C_{FN}(i)+\sum_{FP}C_{FP}(i)\), где \(C_{FN}(i)\approx r_i\), \(C_{FP}(i)=c_I\).



## 6) Лицензия
MIT (см. `LICENSE`).
