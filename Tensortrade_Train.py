import ccxt
import pandas as pd
import requests
from tensortrade.env.default import TradingEnvironment
from tensortrade.environments import TradingEnvironment
from tensortrade.features import FeaturePipeline, SimpleMovingAverage, BollingerBands
from stable_baselines3 import PPO

# 使用CoinGecko API获取市场数据
def get_coin_data(coin_id='bitcoin', vs_currency='usd', days=7):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {'vs_currency': vs_currency, 'days': days}
    response = requests.get(url, params=params)
    data = response.json()
    return data

# 获取数据并转换为DataFrame
data = get_coin_data()
prices = data['prices']  # 获取价格数据
df = pd.DataFrame(prices, columns=['timestamp', 'price'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# 显示数据
print(df.head(10))


# 创建特征管道（技术指标）
feature_pipeline = FeaturePipeline([
    SimpleMovingAverage(window_size=5, feature='price'),  # 简单移动平均线
    BollingerBands(window_size=20, feature='price')  # 布林带
])

# 创建交易环境
environment = TradingEnvironment(
    data=df,  # 使用从CoinGecko获取的数据
    feature_pipeline=feature_pipeline,
    window_size=10,  # 时间窗口大小
    initial_balance=1000,  # 初始资金
    transaction_cost=0.001  # 交易费用
)


# 创建PPO模型，选择MlpPolicy作为模型结构
model = PPO('MlpPolicy', environment, verbose=1)

# 训练模型
model.learn(total_timesteps=100000)


# 评估模型：进行100个时间步的交易
obs = environment.reset()

# 模拟交易过程
for i in range(100):
    action, _states = model.predict(obs)  # 预测当前动作
    obs, rewards, done, info = environment.step(action)  # 执行动作
    if done:
        break  # 交易结束

# 输出最终余额
print(f"Final Balance: {environment.balance}")

