# Integration of Aliyun BaiLian Platform TTS with Home Assistant

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

8. Click the **Configure** button, and enter your BaiLian Platform Token (https://bailian.console.aliyun.com/?apiKey=1#/api-key).

   Voice list: https://help.aliyun.com/zh/model-studio/developer-reference/timbre-list

   After entering the information, click **Submit**.

   ![](https://raw.githubusercontent.com/itning/hass-aliyun_bailian_tts/refs/heads/main/pic/5.png)

9. Click **Settings** > **Voice Assistant** > **Modify Assistant**, and change the engine to `Aliyun BaiLian TTS`.

   ![](https://raw.githubusercontent.com/itning/hass-aliyun_bailian_tts/refs/heads/main/pic/6.png)

10. You're all set!