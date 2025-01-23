
paddlemix `examples` 目录下提供模型的一站式体验，包括模型推理、模型静态图部署、模型预训练，调优等能力。帮助开发者快速了解 PaddleMIX 模型能力与使用方法，降低开发门槛。


## 支持模型

1. 支持一站式训推模型:
> * 多模态理解：ppdocbee、blip2、clip、coca、eva02、evaclip、InternLM-XComposer2、Internvl2、llava、minigpt4、minimonkey、qwen2_vl、qwen_vl、visualglm

2. 支持快速上手体验模型
> * 多模态理解：aira、CogVLM&&CogAgent、deepseek_vl2、GOT_OCR_2_0、imagebind、llava_critic、llava_denseconnector、llava_next、llava_onevision、minicpm-v-2_6、mPLUG_Owl3
> * 多模态理解与生成模型：emu3、janus、showo
> * 开放世界视觉模型：Grouding DINO、YOLO-World、SAM、SAM2
> * 音频生成: audioldm2、diffsinger

3. 支持NPU训练模型：Internvl2、llava

| Model                                           | Model Size                       | Template          |
|-------------------------------------------------| -------------------------------- | ----------------- |
| [YOLO-World](./YOLO-World/)                     | 640M/800M/1280M                  | yolo_world        |
| [aira](./aria/)                                 | 24.9B                            | aira              |
| [audioldm2](./audioldm2/)                       | 346M/712M                        | audioldm2         |
| [blip2](./blip2/)                               | 7B                               | blip2             |
| [clip](./clip)                                  | 2539.57M/1366.68M/986.71M/986.11M/427.62M/149.62M/151.28M | clip              |
| [coca](./coca/)                                 | 253.56M/638.45M/253.56M/638.45M  | coca              |
| [deepseek_vl2](./deepseek_vl2/)                 | 3B/16B/27B                       | deepseek_vl2      |
| [diffsinger](./diffsinger/)                     | 80M                              | diffsinger        |
| [CogVLM && CogAgent](./cogvlm/)                 | 17B                              | cogvlm_cogagent   |
| [emu3](./emu3/)                                 | 8B                               | emu3              |
| [eva02](./eva02/)                               | 6M/22M/86M/304M                  | eva02             |
| [evaclip](./evaclip/)                           | 1.1B/1.3B/149M/428M/4.7B/5.0B    | evaclip           |
| [groundingdino](./groundingdino/)               | 172M/341M                        | groundingdino     |
| [GOT_OCR_2_0](./GOT_OCR_2_0/)                   | 0.6B                             | GOT_OCR_2_0       |
| [imagebind](./imagebind/)                       | 1.2B                             | imagebind         |
| [InternLM-XComposer2](./internlm_xcomposer2/)   | 7B                               | internlm_xcomposer2 |
| [Internvl2](./internvl2/)                       | 1B/2B/8B/26B/40B                 | internvl2         |
| [janus](./janus/)                               | 1.3B                             | janus             |
| [llava](./llava/)                               | 7B/13B                           | llava             |
| [llava_critic](./llava_critic/)                 | 7B                               | llava_critic      |
| [llava_denseconnector](./llava_denseconnector/) | 7B                               | llava_denseconnector |
| [llava_next](./llava_next_interleave/)          | 0.5B/7B                          | llava_next_interleave |
| [llava_onevision](./llava_onevision/)           | 0.5B/2B/7B                       | llava_onevision   |
| [minicpm-v-2_6](./minicpm_v_2_6/)               | 8B                               | minicpm_v_2_6     |
| [minigpt4](./minigpt4/)                         | 7B/13B                           | minigpt4          |
| [minimonkey](./minimonkey/)                     | 2B                               | minimonkey        |
| [mPLUG_Owl3](./mPLUG_Owl3/)                     | 7B                               | mPLUG_Owl3        |
| [ppdocbee](./ppdocbee/)                         | 2B/7B                            | ppdocbee          |
| [qwen2_vl](./qwen2_vl/)                         | 2B/7B/72B                        | qwen2_vl          |
| [qwen_vl](./qwen_vl/)                           | 7B                               | qwen_vl           |
| [sam](./sam/)                                   | 86M/307M/632M                    | sam               |
| [sam2](./sam2/)                                 | 224M                             | sam2              |
| [showo](./showo/)                               | 1.3B                             | showo             |
| [visualglm](./visualglm/)                       | 6B                               | visualglm         |