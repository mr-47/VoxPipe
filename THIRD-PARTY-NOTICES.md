# Third-party notices

**VoxPipe** is MIT-licensed (see `LICENSE`). That license covers **only
our source code**. The pre-trained model weights it downloads are separate works
under their own licenses, reproduced below. The Python dependencies keep their
own licenses; they are not redistributed here.

This file is the authoritative attribution for the models, and for the third-party
server binaries `install.cmd` offers to download on Windows. It is tracked in
git on purpose: `vendor/` is git-ignored, so notices kept only there would
vanish on a fresh clone. `scripts/fetch-models.sh` copies this file next to the
weights.

## Vendored model weights

| File (`vendor/models/`) | Upstream | License | Copyright |
| --- | --- | --- | --- |
| `ggml-silero-v5.1.2.bin` | [ggml-org/whisper-vad](https://huggingface.co/ggml-org/whisper-vad) (Hugging Face), converted from [Silero VAD](https://github.com/snakers4/silero-vad) | MIT | 2020-present Silero Team |
| `diar/sherpa-onnx-pyannote-segmentation-3-0/model.onnx` | [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) release `speaker-segmentation-models`, itself from [pyannote.audio](https://github.com/pyannote/pyannote-audio) | MIT | 2022 CNRS |
| `diar/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` | [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) release `speaker-recongition-models`, itself from [3D-Speaker](https://github.com/modelscope/3D-Speaker) (ModelScope / Alibaba) | Apache-2.0 | see appendix below |
| `diar/sherpa-onnx-pyannote-segmentation-3-0/LICENSE` | copied verbatim out of the segmentation tarball | MIT | 2022 CNRS |

The GGML speech models (180-550 MB) are **not** vendored. If you fetch them
with `scripts/fetch-models.sh turbo`, they come from
[ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp) and are
MIT-licensed, Copyright (c) 2023 Georgi Gerganov and the whisper.cpp authors.

## Adapted source code

`src/voxpipe/format.py` and `src/voxpipe/merge.py` are
adapted from the upstream `transcriber` project (faster-whisper + pyannote,
version 0.2.3), which declares `license = "MIT"` in its `pyproject.toml`. That
upstream ships no `LICENSE` file and names no copyright holder, so no upstream
copyright line can be reproduced here; the MIT terms it declares are applied to
the derived files under our `LICENSE`.

## Python dependencies

Verified from installed package metadata (PEP 639 `License-Expression`):

| Package | License |
| --- | --- |
| `av` (PyAV) | BSD-3-Clause |
| `sherpa-onnx` | Apache-2.0 (per its upstream `LICENSE`) |
| `numpy` | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| `httpx` | BSD-3-Clause |
| `fastapi` | MIT |
| `pydantic` | MIT |
| `starlette` | BSD-3-Clause |
| `uvicorn` | BSD-3-Clause |
| `python-multipart` | Apache-2.0 |
| `pytest` (dev) | MIT |
| `ruff` (dev) | MIT |

Note on PyAV: its wheels bundle FFmpeg shared libraries (`av.libs`). The
bundled `libavcodec` exposes no `--enable-gpl` / `--enable-nonfree` build
flags, so the FFmpeg build appears to be LGPL rather than GPL. If you
redistribute this project, confirm the FFmpeg licensing for the exact wheel you
ship; PyAV's own wheel metadata lives in
`site-packages/av-*.dist-info/licenses/`.

whisper.cpp itself is a separate build, vendored as a relocatable copy in
`vendor/bin/` by `scripts/vendor-server.sh`. It is MIT-licensed,
Copyright (c) 2023 Georgi Gerganov and the whisper.cpp authors.

## Windows server binaries fetched at install time

Nothing in this repository is redistributed here. `install.cmd` offers to
**download** one of these onto the user's machine, which is why they are listed:
the choice is made at install time, and a user should be able to see what the
alternatives are before accepting one. Only the first-party CPU archive is
offered by default on a machine with no GPU.

| Option | Publisher | What it is | License as published |
| --- | --- | --- | --- |
| `whisper-bin-x64.zip` `v1.9.2` | [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp) | Official release asset, CPU build | MIT (upstream project) |
| `whisper.cpp-windows-vulkan.zip` `v1.0.0` | [jerryshell/whisper.cpp-windows-vulkan-bin](https://github.com/jerryshell/whisper.cpp-windows-vulkan-bin) | Third-party rebuild, Vulkan | **None stated** |
| `whisper.cpp-windows-vulkan.zip` `v1.0` | [DomoticX/whisper.cpp-windows-vulkan](https://github.com/DomoticX/whisper.cpp-windows-vulkan) | Third-party rebuild, Vulkan | **None stated** |

The underlying source of all three is whisper.cpp, which is MIT-licensed as
above. That is a statement about the code they are compiled from, **not** a
license grant from whoever compiled them: neither third-party repository
publishes a `LICENSE` file, a `README`, or any SPDX license metadata, and the
binaries are not code-signed. `install.cmd` shows the sha256 of each archive and
verifies it before extracting anything, which is what makes the *download*
verifiable; it cannot make the *publisher* one whose terms we can quote. If that
is not a trade you want to make, take the CPU option, or build whisper.cpp
yourself with `-DGGML_VULKAN=ON` and point `TRANSCRIBER_SERVER_BIN` at the
result.

Neither third-party publisher states which whisper.cpp version was built. The
binaries share 43-44 of 45 MSVC lambda symbol ids with upstream `v1.9.2`, which
is consistent with the same or a very nearby source tree but is not proof, so
the version is recorded nowhere rather than guessed. `install.cmd` therefore pins
them by tag plus hash, not by a version claim.

---

## Model license texts

### Silero VAD - MIT (ggml-silero-v5.1.2.bin)

```
MIT License

Copyright (c) 2020-present Silero Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### pyannote segmentation 3.0 - MIT (model.onnx, and its LICENSE file)

```
MIT License

Copyright (c) 2022 CNRS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 3D-Speaker CAMPPlus - Apache-2.0 (3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx)

The upstream `3D-Speaker/LICENSE` is the unmodified Apache License 2.0. The
appendix retains its placeholder copyright line, which we have left as
published rather than inventing a holder.

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
