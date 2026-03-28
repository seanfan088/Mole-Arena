# Mole Arena Prototype

一个纯 HTML/CSS/JavaScript 的 16:9 网页打地鼠对战原型。

## 原型内容

- 人类 vs AI 双面板对战
- 16:9 横屏布局
- 实时比分、命中率、时间、回合状态
- 游戏时间自定义
- 难度 1-10 自定义
- 开始、暂停、重置
- 单局 / Best of 3 / Best of 5
- 每局结果记录与场次比分
- 人类侧摄像头手部追踪接入第一步：可请求真实摄像头预览，当前仍用鼠标轨迹模拟手部命中
- AI 侧 YOLO 识别接入第二步：加入 synthetic YOLO 检测桥、检测状态面板、候选目标选择和检测框扫描节奏

## 运行

直接在浏览器打开 `index.html` 即可。

## 当前交互

- 点击 `Start` 开始
- 左侧 Human 区优先使用手部识别；识别失败时可继续用鼠标击打
- `Pause` 可暂停/继续
- `Reset` 可重置整场
- `Space` 可开始或暂停
- `R` 可重置

## AI Bridge 协议

当前已开放一个最小 AI 检测输入协议，页面会监听 `window.postMessage`。

### 检测结果消息

```js
window.postMessage({
  type: 'YOLO_DETECTIONS',
  detections: [
    { holeIndex: 2, label: 'mole', confidence: 0.94 },
    { holeIndex: 5, label: 'bomb', confidence: 0.81 }
  ]
}, '*');
```

### 状态消息

```js
window.postMessage({
  type: 'YOLO_BRIDGE_STATUS',
  state: 'external-ready',
  protocol: 'window.postMessage'
}, '*');
```

### 调试页

- `ai-bridge-demo.html` 可作为桥接演示页，向主游戏页发送模拟 YOLO 检测结果

### 本地 HTTP Bridge

项目里新增了两个文件：

- `yolo_bridge_server.py`：本地桥服务，提供 `http://127.0.0.1:8765/api/detections`
- `yolo_bridge_client.js`：页面轮询该接口，并自动转成游戏内部的 `window.postMessage`

服务返回格式示例：

```json
{
  "state": "external-ready",
  "protocol": "http-poll",
  "source": "webcam-yolo:yolov8n.pt",
  "detections": [
    {"holeIndex": 2, "label": "mole", "confidence": 0.94},
    {"holeIndex": 5, "label": "bomb", "confidence": 0.81}
  ]
}
```

### 启动方式

手动桥服务：

```bash
/opt/miniconda3/bin/python3 yolo_bridge_server.py
```

合成检测循环：

```bash
/opt/miniconda3/bin/python3 yolo_bridge_server.py --synthetic
```

摄像头 + YOLO：

```bash
/opt/miniconda3/bin/python3 yolo_bridge_server.py --webcam --model yolov8n.pt
```

如果 8765 被占用，可以换端口，例如：

```bash
/opt/miniconda3/bin/python3 yolo_bridge_server.py --synthetic --port 8777
```

然后在浏览器控制台执行：

```js
localStorage.setItem('mole-arena-yolo-endpoint', 'http://127.0.0.1:8777/api/detections');
location.reload();
```

说明：

- `--webcam` 会打开本机摄像头，使用 YOLO 检测画面目标
- 当前会把检测框中心粗映射到 3x3 洞位
- 如果你的 YOLO 模型类别名里包含 `bomb`，会映射成炸弹；其他类别默认映射成 `mole`

## 下一步建议

1. 接入 MediaPipe Hand Landmarker，把当前摄像头预览和鼠标轨迹升级为真实手部关键点命中
2. 把 current synthetic YOLO bridge 替换为真实 YOLO 推理输入，比如 WebSocket、Worker 或 ONNX/WebGPU
3. 继续补音效、连击反馈、结算弹层和命中回放
