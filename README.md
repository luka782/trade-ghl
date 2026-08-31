# A 股量化研究与回测 MVP

一个本地运行的 A 股研究工作台，包含历史行情缓存、横截面因子分析、等权 Top-N
回测和可视化页面。系统只用于研究和模拟，不包含实盘下单能力。

## 快速启动（Windows PowerShell）

环境要求：

- Python 3.11 或更高版本
- Node.js 20 或更高版本

打开两个 PowerShell 窗口，在项目根目录分别执行：

```powershell
.\start-backend.ps1
```

```powershell
.\start-frontend.ps1
```

首次启动会安装依赖。随后访问：

- Web 页面：<http://localhost:5173>
- API 文档：<http://localhost:8000/docs>
-打不开：
```powershell
cd C:\Users\markguo\股票
Set-ExecutionPolicy -Scope Process Bypass -Force
.\start-frontend.ps1
```
如果 PowerShell 阻止本地脚本，可以仅对当前进程放开：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## 在另一台 Windows 电脑上使用

本仓库只保存源码、测试、启动脚本和依赖清单；本地下载的行情、回测结果和
SQLite 数据库不会上传。克隆到新电脑后需要重新下载所需历史行情，或从备份恢复
数据目录。

1. 安装 [Python 3.11 或更高版本](https://www.python.org/downloads/) 与
   [Node.js 20 或更高版本](https://nodejs.org/)；安装完成后重新打开 PowerShell。
2. 克隆项目并进入目录：

```powershell
git clone https://github.com/luka782/trade-ghl.git
Set-Location trade-ghl
```

3. 分别打开两个 PowerShell 窗口，均进入项目根目录。在第一个窗口启动后端：

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\start-backend.ps1
```

4. 在第二个窗口启动前端：

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\start-frontend.ps1
```

5. 在浏览器访问 <http://localhost:5173>。首次运行会自动创建 Python 虚拟环境、
   安装后端和前端依赖；随后请在“数据管理”页面下载需要研究的历史数据。

## 使用流程

1. 在“数据管理”中选择至少 20 只股票，下载两年以上的前复权或不复权日线。
2. 在“因子研究”中选择 `20 日动量`，设置未来收益周期并运行分析。
3. 查看 IC、RankIC、分组净值、因子覆盖率和换手率。
4. 在“策略回测”中配置 Top 10、月度调仓、费用和基准后运行回测。
5. 在“任务结果”中重新打开历史回测。

默认示例股票池只为快速体验，不代表投资建议，也不代表任何历史指数成分。

## 项目结构

```text
backend/
  app/
    data/          # 数据源接口与 AkShare 实现
    factors/       # 因子、预处理与有效性评价
    backtest/      # 调仓、成交约束与绩效统计
    timing/        # 单标的择时状态机与技术指标
    validation/    # Walk-Forward 与过拟合诊断
    main.py        # FastAPI
    storage.py     # Parquet 与 SQLite
  tests/
  user_factors/   # 本地 Python 因子插件
frontend/
  src/
backend/Dockerfile
frontend/Dockerfile
docker-compose.yml
Caddyfile
start-backend.ps1
start-frontend.ps1
```

## 回测口径

- 因子只使用 T 日收盘时已知数据。
- 内置和自定义因子会在多个历史截面上做“删除未来行后重算”检查，常见的
  `shift(-n)` 未来函数会直接终止分析或回测。
- 信号在 T 日收盘后形成，交易按 T+1 日收盘价模拟；新持仓从交易完成后的下一交易日
  开始贡献收益。这一口径比“T+1 开盘成交”更保守，也避免使用不可获得的成交价格。
- 前复权价格用于因子和收益核算；停牌、成交价格限制和分币舍入使用单独缓存的不复权
  行情。下载前复权数据时会自动同时更新这份成交约束数据。
- 手续费作用于买卖双边并执行每笔 5 元最低佣金；印花税只作用于卖出，默认按
  2023-08-28 政策切换日期使用历史税率；滑点作用于双边。
- API 可将 `historical_stamp_duty` 设为 `false`，此时整个区间使用用户填写的固定税率。
- 成交量为零或缺失视为停牌；封死涨停不能买入，封死跌停不能卖出。
- 涨跌停价按不复权参考价四舍五入到 0.01 元；创业板在 2020-08-24 前后分别按
  10%/20% 处理。
- 持仓连续缺少估值行情超过 20 个市场交易日时终止回测，避免退市证券永久沿用旧价。

## 数据与研究限制

- AkShare 是第三方免费数据接口，上游限流、字段调整或临时不可用会导致下载失败。
- 个股行情优先使用东方财富接口，失败时回退到新浪接口；统一后的股票成交量单位为“股”。
- 前复权序列可能随着后续公司行为发生变化。严谨复现实验应保存数据快照。
  当缓存向未来扩展时，系统会重新获取完整前复权区间，而不是直接拼接不同缩放口径。
- 默认股票池是当前用户选择并成功下载的股票，存在幸存者偏差。系统不会把当前指数
  成分股伪装成历史成分股。
- AkShare 日线接口不包含历史时点 PE/PB。内置 PB 因子只有在输入数据具有可靠的
  point-in-time `pb` 字段时才可运行，否则会明确报告不可用。
- AkShare 日线也不提供可靠的历史 ST 区间；系统会将其标记为“未知”并在结果中警告，
  而不会伪装成已知非 ST。未知时仍按普通板块限制计算。
- 没有逐笔委托簿时，收盘价恰好处于涨跌停价会被保守地视为无法成交。
- 上市/重新上市初期无涨跌停窗口和买入 100 股整数手约束尚未建模。
- 退市、长期停牌后的资产估值、现金分红税费、配股和逐笔成交冲击尚未达到生产级精度。
- 本项目输出仅供软件与量化研究学习，不构成投资建议。

## 测试

后端测试使用合成行情，不访问外网：

```powershell
Set-Location backend
.\.venv\Scripts\python.exe -m pytest
```

前端类型检查与生产构建：

```powershell
npm --prefix frontend run build
```

## 添加 Python 因子

复制 `backend/user_factors/_example.py`，将新文件改成不以下划线开头的名称，然后修改
`FactorMetadata` 和 `compute()`。文件导出 `FACTOR` 或 `FACTORS` 后，重启后端即可在
因子列表中使用。详细约束见 `backend/user_factors/README.md`。

内置研究因子：

- `momentum_20`：20日动量（正向）
- `reversal_5`：5日反转（正向）
- `volatility_20`：20日波动率（负向）
- `volume_change_20`：20日成交量变化率（正向）
- `ma_bias_20`：20日均线偏离度（正向）
- `price_position_60`：60日价格位置（正向）
- `downside_volatility_20`：20日下行波动率（负向）
- `amihud_20`：20日非流动性（负向）

负向因子保留原始 IC，同时使用方向调整后的分数进行分组和 Top-N 排名。

## 批量研究

在 `backend` 目录运行：

```powershell
python -m scripts.build_universe
python -m scripts.run_factor_research
python -m scripts.run_etf_research
```

第一条命令扫描当前全部沪深主板普通A股，按照最近60个完整交易日平均成交额各选
50只，并下载前复权研究行情和不复权成交约束行情。第二条命令生成
全样本、样本内及最近12个完整月样本外的因子与回测报告。结果保存在
`backend/reports/<最新完整交易日>/`。

第三条命令下载20只境内股票型ETF，并在免印花税、T+1、Top 5月度调仓口径下运行
相同的8个因子测试。ETF报告保存在 `backend/reports/<日期>/etfs/`。

常用数据接口还包括：

- `GET /api/data/calendar`：A 股交易日历
- `GET /api/data/adjustment-factors/{symbol}`：前复权因子
- `GET /api/data/status`：本地 Parquet 数据状态
- `GET /api/research/universe`：最近生成的100只主板研究股票池
- `GET /api/research/etfs`：20只代表性ETF测试集

## 配置

后端支持以下环境变量：

- `QUANT_DATA_DIR`：Parquet 数据目录
- `QUANT_DB_PATH`：SQLite 文件路径
- `QUANT_USER_FACTOR_DIR`：自定义 Python 因子目录

前端可在 `frontend/.env.local` 中配置：

```text
VITE_API_URL=http://localhost:8000/api
```

## 阿里云 ECS 生产部署

以下步骤适用于 Ubuntu 22.04、Docker Compose 和阿里云 ECS。生产容器包括：

- `backend`：FastAPI，只在 Docker 内部监听 `8000`；
- `frontend`：Nginx 提供已构建的 React 静态文件，只在 Docker 内部监听 `80`；
- `caddy`：唯一公网入口，监听宿主机 `80/443`，将 `/api/*` 转发到后端。

Parquet、SQLite、报告和用户因子全部保存在 ECS 的 `/data/quant`，重新构建或删除
容器不会删除该目录。GitHub 只保存代码，不保存运行数据。

### 1. ECS 与安全组

建议至少使用 2 核 4GB、Ubuntu 22.04 和 50GB 云盘。安全组入方向只开放：

- TCP `22`：建议只允许自己的公网 IP；
- TCP `80`：首次 IP 测试使用；
- TCP/UDP `443`：配置域名 HTTPS 后使用。

不要开放 `8000` 或 `5173`。本项目当前没有用户登录系统，通过公网 IP 测试时，
建议暂时将 `80` 端口来源限制为自己的公网 IP；长期公网使用前应增加身份认证。

### 2. 安装 Docker 并克隆项目

通过 SSH 登录 ECS 后执行：

```bash
apt update
apt install -y git curl ca-certificates
curl -fsSL https://get.docker.com | sh
systemctl enable --now docker

git clone https://github.com/luka782/trade-ghl.git /opt/trade-ghl
cd /opt/trade-ghl
```

### 3. 创建持久化目录和环境配置

```bash
mkdir -p /data/quant/parquet
mkdir -p /data/quant/reports
mkdir -p /data/quant/user_factors
mkdir -p /data/quant/backups

cp .env.example .env
chmod 600 .env
```

首次通过 ECS 公网 IP 测试时，`.env` 保持：

```text
SITE_ADDRESS=:80
QUANT_HOST_DATA_ROOT=/data/quant
```

真实 `.env` 已被 Git 忽略，不要提交其中的域名或其他私密配置。

### 4. 构建、启动和检查

```bash
cd /opt/trade-ghl
docker compose config
docker compose up -d --build
docker compose ps
docker compose logs --tail=100
curl http://127.0.0.1/api/health
```

随后访问 `http://ECS公网IP`。第一次需要在“数据管理”页面下载历史行情，下载结果
会写入 `/data/quant/parquet`，回测记录会写入 `/data/quant/quant.db`。

### 5. 配置域名与 HTTPS

在域名 DNS 中添加 A 记录，将例如 `quant.example.com` 指向 ECS 公网 IP。解析生效后
修改服务器的 `/opt/trade-ghl/.env`：

```text
SITE_ADDRESS=quant.example.com
QUANT_HOST_DATA_ROOT=/data/quant
```

然后重新加载 Caddy：

```bash
cd /opt/trade-ghl
docker compose up -d
docker compose logs --tail=100 caddy
```

Caddy 会自动申请和续期 HTTPS 证书。确认 `https://quant.example.com` 可访问后，
可以在安全组中取消不必要的公网 HTTP 测试规则；保留 `80` 时 Caddy 会自动跳转 HTTPS。

### 6. 更新线上版本

先在开发电脑完成测试并推送 GitHub，再在 ECS 执行：

```bash
cd /opt/trade-ghl
git pull --ff-only origin main
docker compose up -d --build --remove-orphans
docker compose ps
curl http://127.0.0.1/api/health
```

代码更新不会删除 `/data/quant`。若要回滚，先从 `git log --oneline` 找到稳定提交，
切换到该提交后重新构建：

```bash
git switch --detach <稳定提交号>
docker compose up -d --build
```

回到最新版：

```bash
git switch main
git pull --ff-only origin main
docker compose up -d --build
```

### 7. 备份和恢复

为避免备份到正在写入的 SQLite 文件，最简单可靠的个人部署方案是短暂停止后端，
再压缩整个数据目录：

```bash
cd /opt/trade-ghl
docker compose stop backend
tar -C /data -czf "/data/quant/backups/quant-$(date +%F-%H%M%S).tar.gz" \
  --exclude='quant/backups' quant
docker compose start backend
```

将备份包下载到本地或上传到阿里云 OSS，才能防止误删 ECS/云盘导致全部副本丢失。
OSS AccessKey 应配置在服务器安全的凭据文件或实例 RAM 角色中，不要写入仓库。

恢复前先停止后端，并确认备份文件路径：

```bash
cd /opt/trade-ghl
docker compose stop backend
mv /data/quant "/data/quant.before-restore-$(date +%F-%H%M%S)"
mkdir -p /data/quant
tar -C /data -xzf /安全位置/quant-YYYY-MM-DD-HHMMSS.tar.gz
docker compose start backend
docker compose ps
```

恢复操作会替换行情和 SQLite 状态，执行前必须保留当前 `/data/quant` 的副本。
