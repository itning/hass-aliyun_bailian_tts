<h3 align="center">Integration of Aliyun BaiLian Platform TTS with Home Assistant</h3>
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

## Introduction
The Home Assistant TTS (Text-to-Speech) feature leverages the voice synthesis capabilities of the **CosyVoice** large language model from the [Aliyun BaiLian Platform](https://bailian.console.aliyun.com/).

Currently, only **CosyVoice** is supported. For more information, please refer to the [Aliyun Documentation](https://help.aliyun.com/zh/model-studio/developer-reference/cosyvoice-large-model-for-speech-synthesis/).

## Installation

### Installation via HACS

1. Go to the HACS panel, click the menu button in the top-right corner, and add a custom integration.

   ![](https://raw.githubusercontent.com/itning/hass-aliyun_bailian_tts/refs/heads/main/pic/1.png)

2. In the pop-up panel:

   - **Repository**: Enter `https://github.com/itning/hass-aliyun_bailian_tts.git`
   - **Type**: Select `Integration`

   ![](https://raw.githubusercontent.com/itning/hass-aliyun_bailian_tts/refs/heads/main/pic/2.png)

3. Click the **ADD** button to save.

4. In HACS, search for the newly added integration: `Aliyun BaiLian TTS`. Click the **Download** button on the right side of the menu.

   ![](https://raw.githubusercontent.com/itning/hass-aliyun_bailian_tts/refs/heads/main/pic/3.png)

5. After the download is complete, restart Home Assistant as prompted.

6. After restarting, go to **Settings** > **Devices & Services**, then click the button in the bottom-right corner to add an integration. Search for `Aliyun BaiLian TTS` and add it.

   ![](https://raw.githubusercontent.com/itning/hass-aliyun_bailian_tts/refs/heads/main/pic/4.png)

7. Click on the added `Aliyun BaiLian TTS` to enter the configuration page.

8. Click the **Configure** button, and enter your BaiLian Platform Token (https://bailian.console.aliyun.com/?tab=model#/api-key).

   Voice list: https://help.aliyun.com/zh/model-studio/developer-reference/timbre-list

   After entering the information, click **Submit**.

   ![](https://raw.githubusercontent.com/itning/hass-aliyun_bailian_tts/refs/heads/main/pic/5.png)

9. Click **Settings** > **Voice Assistant** > **Modify Assistant**, and change the engine to `Aliyun BaiLian TTS`.

   ![](https://raw.githubusercontent.com/itning/hass-aliyun_bailian_tts/refs/heads/main/pic/6.png)

10. You're all set!

## Conclusion

Most of the code in this project was generated by the large language model [qwen-max-2025-01-25](https://bailian.console.aliyun.com/model-market/detail/qwen-max-2025-01-25#/model-market/detail/qwen-max-2025-01-25).

## Acknowledgments

![JetBrains Logo (Main) logo](https://resources.jetbrains.com/storage/products/company/brand/logos/jb_beam.svg)
