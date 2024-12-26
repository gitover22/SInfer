### SInfer + Prometheus/Grafana 

这是一个简单的示例，展示了如何将 SInfer 的指标日志连接到 Prometheus/Grafana 堆栈。在该示例中，我们通过 Docker 启动 Prometheus 和 Grafana。有关其他方法，可以参考 [Prometheus](https://prometheus.io/) 和 [Grafana](https://grafana.com/) 的官方网站。

---

### **安装依赖**
- [`docker`](https://docs.docker.com/engine/install/)
- [`docker compose`](https://docs.docker.com/compose/install/linux/#install-using-the-repository)

---

### **启动步骤**

#### **1. 启用 SInfer 的 Prometheus 指标日志**
在 OpenAI 兼容的服务器中，默认启用了 Prometheus 的指标日志。可以通过以下命令启动：

```bash
python3 -m sinfer.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-v0.1 \
    --max-model-len 2048 \
    --disable-log-requests
```

#### **2. 启动 Prometheus 和 Grafana 服务**
通过 `docker compose` 启动 Prometheus 和 Grafana 服务：
```bash
docker compose up
```

#### **3. 提交测试请求**
可以向服务器提交一些示例请求，使用以下命令下载测试数据集并运行测试：

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

python3 ../../benchmarks/benchmark_serving.py \
    --model mistralai/Mistral-7B-v0.1 \
    --tokenizer mistralai/Mistral-7B-v0.1 \
    --endpoint /v1/completions \
    --dataset ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate 3.0
```

在浏览器中访问 [`http://localhost:8000/metrics`](http://localhost:8000/metrics)，可以查看 SInfer 暴露的原始 Prometheus 指标数据。

---

### **Grafana 仪表板**

#### **1. 访问 Grafana**
在浏览器中访问 [`http://localhost:3000`](http://localhost:3000)。使用默认的用户名（`admin`）和密码（`admin`）登录。

#### **2. 添加 Prometheus 数据源**
前往 [`http://localhost:3000/connections/datasources/new`](http://localhost:3000/connections/datasources/new)，选择 Prometheus。

在 Prometheus 配置页面中，需要在 “Connection” 部分添加 `Prometheus Server URL`。在此设置中，Grafana 和 Prometheus 运行在不同的容器中，但 Docker 为每个容器创建了 DNS 名称，因此可以直接使用 `http://prometheus:9090`。

点击 `Save & Test`。如果成功，将显示绿色的提示：“Successfully queried the Prometheus API”。

#### **3. 导入仪表板**
前往 [`http://localhost:3000/dashboard/import`](http://localhost:3000/dashboard/import)，上传 `grafana.json` 文件，并选择 `prometheus` 数据源。

完成后，您应该能看到一个类似以下内容的仪表板：

![Grafana 仪表板](https://i.imgur.com/R2vH9VW.png)

---

通过上述步骤，您可以在 Prometheus 和 Grafana 中监控 SInfer 的性能指标和推理数据。

