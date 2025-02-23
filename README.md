# 阿里云百炼平台TTS与Home Assistant集成

[ENGLISH README](https://github.com/itning/hass-aliyun_bailian_tts/blob/main/README-en.md)

## 简介
Home Assistant TTS发音使用阿里云[百炼平台](https://bailian.console.aliyun.com/)的语音合成CosyVoice大模型

目前只支持`CosyVoice` 具体可以查看[阿里云文档](https://help.aliyun.com/zh/model-studio/developer-reference/cosyvoice-large-model-for-speech-synthesis/
)

## 安装

### 通过HACS安装

1.前往HACS面板，点击右上角菜单，添加自定义集成 

![](https://raw.githubusercontent.com/itning/hass-aliyun_bailian_tts/refs/heads/main/pic/1.png)

2.在弹出的面板中，

Repository填写`https://github.com/itning/hass-aliyun_bailian_tts.git`

type选择：`Integration` 

![](https://raw.githubusercontent.com/itning/hass-aliyun_bailian_tts/refs/heads/main/pic/2.png)

3.点击ADD按钮进行保存。

4.在HACS中搜索刚刚添加的集成：`Aliyun BaiLian TTS` 并在右侧菜单中点击`Download`按钮

![](https://raw.githubusercontent.com/itning/hass-aliyun_bailian_tts/refs/heads/main/pic/3.png)

5.下载完成后按照提示重启Home Assistant

6.重启后在设置中点击设备与服务，然后点击右下角按钮添加集成，搜索`Aliyun BaiLian TTS` 并添加。

![](https://raw.githubusercontent.com/itning/hass-aliyun_bailian_tts/refs/heads/main/pic/4.png)

7.点击添加的`Aliyun BaiLian TTS`进入配置页面

8.点击配置按钮，输入百炼平台Token （https://bailian.console.aliyun.com/?apiKey=1#/api-key）

音色（Voice）列表：https://help.aliyun.com/zh/model-studio/developer-reference/timbre-list

输入完成后，点击提交。

![](https://raw.githubusercontent.com/itning/hass-aliyun_bailian_tts/refs/heads/main/pic/5.png)

9.点击设置，语音助手，修改助手，将引擎改成`Aliyun BaiLian TTS`

![](https://raw.githubusercontent.com/itning/hass-aliyun_bailian_tts/refs/heads/main/pic/6.png)

10.完成。
