# Ethics, Cultural Representation & AI Safety

## Project Overview & Cultural Narrative Scope
*Heritage-Short* explores a deeply personal and culturally resonant Ethiopian tradition: roasting green coffee beans (*buna qoloc’ho*) over glowing charcoal (*kesel*) on rainy Kiremt afternoons in Addis Ababa. The central narrative is anchored by maternal wisdom through the classic Amharic proverb:
> **"ቀስ በቀስ እንቁላል በእግሩ ይሄዳል"**  
> *(Slowly, slowly, an egg learns to walk)*

Because this short film portrays family memories, domestic intimacy, and shared cultural heritage, this document establishes ethical standards, consent protocols, license audits, and provenance guidelines before any media generation.

---

## Consent Record & Likeness Protocol

### Subject & Likeness Documentation
- **Subject / Character**: The Mother (*Emaye*).
- **Representation Strategy**: **Stylized Artistic Representation**.
  - No real photographic facial scans, biometric landmarks, or high-dimensional identity embeddings of living family members are fed into any model or prompt.
  - Visual prompts depict a dignified, painterly, stylized Ethiopian maternal figure in traditional Habesha Kemis, avoiding deepfake replication.
- **Informed Family Consent Log**:
  - **Date Recorded**: November 25, 2026.
  - **Participants**: Story author & Mother.
  - **Consultation Scope**: Script narrative, Amharic proverb interpretation, vocal cadence tone, artistic visual direction, and intended distribution channels.
  - **Approved Distribution Channels**: Non-commercial social media sharing (Instagram Reels, TikTok, YouTube Shorts, WhatsApp family/community groups).
  - **Revocation Right**: The family retains the right to withdraw authorization or request modifications at any production milestone prior to final export.

### Privacy Safeguards & Data Minimization
- **Strict Prohibition on Private Ingestion**: No personal family photographs, unreleased home audio recordings, physical home addresses, or private identifying markers may be uploaded to third-party or free-tier cloud endpoints.
- **Prompt Sanitization**: All prompts use descriptive archetype language (e.g., *"an Ethiopian mother smiling warmly with weathered hands turning coffee beans over charcoal in an Addis Ababa veranda"*), completely free of personal legal names or private descriptors.

---

## Model & Output License Table

Every model and tool utilized across the multi-stage pipeline is audited for licensing terms, output ownership, commercial/non-commercial allowances, and training data privacy implications:

| Model / Tool | Pipeline Stage | Provider / Host | Underlying License | Creative & Commercial Rights | Data Privacy & Training Policy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gemini 3.8 Flash** | Storyboard & Prompt Expansion | Google Cloud / GenAI API | Google Generative AI Additional Terms of Service | Permitted for user-generated content; user owns input/output per Google TOS. | Enterprise/Paid endpoints do not log for training. **Note on Free Tier:** Free endpoints may log prompts for product improvement—no confidential family data permitted in prompts. |
| **Gemini 3.8 Flash TTS** | Voice Narration & Speech Synthesis | Google Cloud / GenAI API | Google Generative AI Additional Terms of Service | Audio outputs retain broad usage rights; synthesized speech must not mimic real individuals without explicit voice rights. | Audio prompts and synthesized output handled under Google API data retention rules. |
| **FLUX.2 klein 4B / FLUX.1 schnell** | Keyframe Anchor Generation | Black Forest Labs / Cloudflare Workers AI | Apache 2.0 (FLUX.1 schnell & FLUX.2 klein open weights) | Full commercial & non-commercial rights under Apache 2.0. Outputs are royalty-free. | Cloudflare Workers AI ephemeral inference: inputs and generated images are processed in-memory and not ingested for training. |
| **Wan 2.2 TI2V-5B** | Text/Image-to-Video Shot Motion | Wan-Video / Open-Weights | Apache 2.0 | Apache 2.0 open-source permissive license; full creative autonomy over generated video frames. | Self-hosted or serverless GPU: zero persistent telemetry or model retraining on inputs. |
| **ACE-Step 1.5** | Cinematic Background Score (Masenqo / Krar / Strings) | ACE Music Foundation | MIT License | Fully permissive under MIT; unrestricted modification, redistribution, and broadcast. | Local / self-hosted execution; entirely isolated from cloud telemetry. |
| **FFmpeg** | Conformance, Muxing & Assembly | FFmpeg Project | LGPL v2.1+ / GPL v2+ | Permissive binary redistribution and media processing for any commercial or non-commercial short film. | Local CLI utility; 100% offline and private. |

---

## AI Labelling & Provenance Plan

To promote transparency, combat deception in synthetic media, and preserve artistic integrity, *Heritage-Short* adheres to a multi-tiered disclosure protocol:

### 1. Platform-Level AI Disclosure Toggles
When uploading the finished film to social platforms, the mandatory synthetic media flags must be enabled:
- **Instagram Reels**: Activate the **"AI info"** label (*"Altered or synthetic media"*).
- **YouTube Shorts**: Check **"Yes"** for *"Altered content: Sound or visuals were significantly edited or digitally generated"*.
- **TikTok**: Toggle **"AI-generated content"** setting prior to publishing.

### 2. Mandatory 2.5-Second Visible AI End Card
The final assembly timeline will conclude with a mandatory 2.5-second title card styled consistently with the film's typography:
```text
+-------------------------------------------------------------+
|                                                             |
|                      HERITAGE SHORT                         |
|                                                             |
|           Created with AI-Assisted Storytelling             |
|                                                             |
|   Visuals: FLUX & Wan 2.2 | Narration: Gemini Speech        |
|   Cultural Story & Direction: Human Author & Family Memory  |
|                                                             |
+-------------------------------------------------------------+
```

### 3. Preserving Metadata & Digital Watermarks
- **C2PA & SynthID Compliance**: Embedded digital signatures, invisible SynthID watermarks from Gemini, and C2PA provenance headers must not be stripped or masked during FFmpeg re-encoding.
- The pipeline's final assembly script will preserve container metadata tagging the video as AI-assisted documentary storytelling.

### 4. Zero-Cost Free-Tier Endpoint Hygiene
- Free-tier APIs (e.g., Cloudflare Workers AI free credits, Google AI Studio free tier) are utilized exclusively with synthetic, generalized, and sanitized descriptions.
- Never upload raw recordings or private media files to unsecured free endpoints.

---

## Cultural Authenticity & Respectful Representation

1. **Linguistic Precision (Amharic)**:
   - All Amharic voice scripts, idiomatic phrasing, and Fidel typography are validated with native speakers to prevent machine-translation awkwardness.
   - The proverb *"ቀስ በቀስ እንቁላል በእግሩ ይሄዳል"* is treated as a philosophical lesson on mindfulness, patience, and generational transfer.

2. **Accurate Portrayal of the Coffee Ritual (*Buna Tetu*)**:
   - Visual and audio cues must faithfully represent authentic Ethiopian coffee traditions: the green beans (*buna*), the clay stove (*furnello*), the blackened iron pan (*biret mitad*), the soothing aroma (*c’hees*), and the pouring into delicate *sini* cups.
   - Avoid generic "exoticized" tropes; the setting is an authentic Addis Ababa household during the cool, reflective Kiremt rainy season.
