<h3 align="center">阿里云百炼平台TTS与Home Assistant集成</h3>
<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/itning/hass-aliyun_bailian_tts.svg?style=social&label=Stars)](https://github.com/itning/hass-aliyun_bailian_tts/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/itning/hass-aliyun_bailian_tts.svg?style=social&label=Fork)](https://github.com/itning/hass-aliyun_bailian_tts/network/members)
[![GitHub watchers](https://img.shields.io/github/watchers/itning/hass-aliyun_bailian_tts.svg?style=social&label=Watch)](https://github.com/itning/hass-aliyun_bailian_tts/watchers)
[![GitHub followers](https://img.shields.io/github/followers/itning.svg?style=social&label=Follow)](https://github.com/itning?tab=followers)


</div>

<div align="center">

[![GitHub issues](https://img.shields.io/github/issues/itning/hass-aliyun_bailian_tts.svg)](https://github.com/itning/hass-aliyun_bailian_tts/issues)
[![GitHub license](https://img.shields.io/github/license/itning/hass-aliyun_bailian_tts.svg)](https://github.com/itning/hass-aliyun_bailian_tts/blob/master/LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/itning/hass-aliyun_bailian_tts.svg)](https://github.com/itning/hass-aliyun_bailian_tts/commits)
[![GitHub repo size in bytes](https://img.shields.io/github/repo-size/itning/hass-aliyun_bailian_tts.svg)](https://github.com/itning/hass-aliyun_bailian_tts)
[![Hits](https://hitcount.itning.com?u=itning&r=hass-aliyun_bailian_tts)](https://github.com/itning/hit-count)

</div>

---

[ENGLISH README](https://github.com/itning/hass-aliyun_bailian_tts/blob/main/README-en.md)

## 简介
Home Assistant TTS发音使用阿里云[百炼平台](https://bailian.console.aliyun.com/)的语音合成大模型

目前支持以下两种语音合成模型：
1. `CosyVoice` - 语音合成CosyVoice大模型，具体可以查看[阿里云文档](https://help.aliyun.com/zh/model-studio/developer-reference/cosyvoice-large-model-for-speech-synthesis/)
2. `Qwen3-TTS` `Qwen-TTS` - 通义千问系列的语音合成模型，支持输入中文、英文、中英混合的文本，并流式输出音频，具体可以查看[阿里云文档](https://help.aliyun.com/zh/model-studio/qwen-tts)

## 安装

### 通过HACS安装

1. [![Open your Home Assistant instance and open the hass-aliyun_bailian_tts integration inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=itning&repository=hass-aliyun_bailian_tts&category=integration) 前往HACS面板，点击右上角菜单，添加自定义集成 

   ![](https://raw.githubusercontent.com/itning/hass-aliyun_bailian_tts/refs/heads/main/pic/1.png)

2. 在弹出的面板中，

   Repository填写`https://github.com/itning/hass-aliyun_bailian_tts.git`

   type选择：`Integration` 

   ![](https://raw.githubusercontent.com/itning/hass-aliyun_bailian_tts/refs/heads/main/pic/2.png)

3. 点击ADD按钮进行保存。

4. 在HACS中搜索刚刚添加的集成：`Aliyun BaiLian TTS` 并在右侧菜单中点击`Download`按钮

   ![](https://raw.githubusercontent.com/itning/hass-aliyun_bailian_tts/refs/heads/main/pic/3.png)

5. 下载完成后按照提示重启Home Assistant

6. 重启后在设置中点击设备与服务，然后点击右下角按钮添加集成，搜索`Aliyun BaiLian TTS` 并添加。

   ![](https://raw.githubusercontent.com/itning/hass-aliyun_bailian_tts/refs/heads/main/pic/4.png)

7. 点击添加的`Aliyun BaiLian TTS`进入配置页面

8. 点击配置按钮，输入百炼平台Token （https://bailian.console.aliyun.com/?tab=model#/api-key）

   音色（Voice）列表：
   - CosyVoice: https://help.aliyun.com/zh/model-studio/text-to-speech
   - Qwen-TTS: https://help.aliyun.com/zh/model-studio/qwen-tts
   
   模型（Model）列表：
   - CosyVoice模型：`cosyvoice-v1`等
   - Qwen-TTS模型：`qwen-tts`、`qwen-tts-latest`、`qwen-tts-2025-05-22`、`qwen-tts-2025-04-10`
   - Qwen3-TTS模型：`qwen3-tts-flash`、`qwen3-tts-flash-2025-09-18`
   - 实时语音模型：`qwen-tts-realtime`...
   
   输入完成后，点击提交。

   ![](https://raw.githubusercontent.com/itning/hass-aliyun_bailian_tts/refs/heads/main/pic/5.png)

9. 点击设置，语音助手，修改助手，将引擎改成`Aliyun BaiLian TTS`

   ![](https://raw.githubusercontent.com/itning/hass-aliyun_bailian_tts/refs/heads/main/pic/6.png)

10. 完成。

## 结尾

此项目前期大部分代码是由大模型[qwen-max-2025-01-25](https://bailian.console.aliyun.com/model-market/detail/qwen-max-2025-01-25#/model-market/detail/qwen-max-2025-01-25)生成的，后期使用Claude Sonnet 4。

## 感谢

![JetBrains Logo (Main) logo](https://resources.jetbrains.com/storage/products/company/brand/logos/jb_beam.svg)
