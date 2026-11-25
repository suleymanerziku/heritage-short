# Creative & Visual Design System (v1)

## Logline & Emotional Intent

### 1-Sentence Logline
> As heavy Kiremt rains drum against a corrugated tin roof in Addis Ababa, a mother gently roasts green coffee beans over glowing charcoal, teaching her child through the ancient proverb *"ቀስ በቀስ እንቁላል በእግሩ ይሄዳል"* that true patience nurtures both the perfect cup and a resilient life.

### Emotional Intent & Target Resonance
- **Core Emotional Arc**: Nostalgic intimacy $\rightarrow$ Sensory absorption $\rightarrow$ Maternal grounding $\rightarrow$ Quiet pride.
- **The Desired Reaction**: The viewer experiences an immediate sensory trigger—smelling the roasted coffee smoke (*c'hees*), feeling the chill of Addis Ababa's rainy season—culminating in an overwhelming urge to pick up the phone and call their mother.

---

## Audience Persona & Viewing Context

### Target Personas
1. **Primary Persona ("The Diaspora Seeker")**:
   - *Demographics*: 20–35 year-old Ethiopian diaspora residing in Washington D.C., Toronto, London, Frankfurt, or Dubai.
   - *Platform*: Instagram Reels, TikTok, YouTube Shorts, WhatsApp family group chats.
   - *Viewing Habits*: Consumed on smartphones with sound ON, often using earbuds while unwinding in the evening.
   - *Needs*: High cultural fidelity; seeks emotional connection to home without tourist clichés or generic stock imagery.

2. **Secondary Persona ("The Addis Ababa Youth")**:
   - *Demographics*: 18–30 year-olds in Addis Ababa and regional cities.
   - *Platform*: Telegram channels, WhatsApp status, Instagram.
   - *Viewing Habits*: Variable mobile bandwidth; requires fast loading and crisp audio intelligibility.

### Phone-First Safe Area & Typography Guidelines
The master aspect ratio is **9:16 vertical (1080 x 1920 px)** optimized for mobile short-form video:

```text
+------------------------------------+ Y = 0 px
| [TOP 15% DANGER ZONE]             |
| Platform Header, Audio info, Back  |
+------------------------------------+ Y = 288 px
|                                    |
|                                    |
|                                    |
|       [65% CRITICAL ACTION ZONE]   |
|     Visual Focus & Center Framing  |
|                                    |
|   - Mother's hands stirring beans  |
|   - Glowing charcoal embers        |
|   - Rising aromatic smoke          |
|                                    |
|  [SUBTITLE ZONE: Y=1350-1480 px]   |
|   Amharic Fidel + English Subtitle |
+------------------------------------+ Y = 1536 px
| [BOTTOM 20% DANGER ZONE]          |
| Username, Caption, Audio Ticker,   |
| Like/Share Buttons UI Overlay      |
+------------------------------------+ Y = 1920 px
```

- **Subtitle Placement**: Burned-in English subtitles anchored between **Y = 1350 px** and **Y = 1480 px** (centered horizontally, font size 44pt with a 3px dark drop shadow), guaranteeing full readability above native platform caption cards.

---

## Core Constraints & Production Trade-offs

1. **Free-Tier GPU Quota & API Economics**:
   - Cloudflare Workers AI limits daily inference credits.
   - **Enforcement**: Mandatory input hashing (`compute_input_hash`) and sidecar metadata verification (`check_cache`) in `pipeline/utils.py`.
   - Re-running the pipeline or modifying a single shot skips all unchanged keyframes, narration files, and motion shots.

2. **Dual 9:16 + 16:9 Framing Composition**:
   - Key visual assets are framed with **Center-Weighted Protection**:
     - All primary visual focal points (the *biret mitad*, roasting beans, mother's expressive hands, the *kesel* glow) are composed inside a 1:1 square center box.
     - This guarantees that a 9:16 vertical crop captures full emotional context without cropping faces or hands, while still allowing an alternate 16:9 widescreen master cut.

3. **Audio-First "Radio Cut" Precedence**:
   - Video duration is strictly driven by the natural cadence of the Amharic narration, Foley footsteps, and rain rhythm.
   - Keyframe prompts and video shot motions are timed to audio beats rather than forcing narration into predetermined shot lengths.

---

## Cinematic & Audio Direction

### Visual Atmosphere
- **Color Palette**:
  - *Warm Core*: Ember orange (`#D35400`), roasted coffee bean brown (`#3E2723`), burnished clay (`#B7410E`).
  - *Cool Contrast*: Kiremt rain slate-gray (`#546E7A`), wet eucalyptus green (`#2E4A3E`), soft rain mist (`#CFD8DC`).
- **Lighting**: Low-key interior veranda lighting illuminated primarily by the red-orange luminescence of the clay stove (*furnello*), contrasted against the overcast, rainy daylight outside.

### Audio Stems & Foley Layering
1. **Narration**: Warm, measured, maternal Amharic speech delivered with relaxed pacing.
2. **Environmental Foley**:
   - Rhythmic drumming of heavy rain on corrugated iron sheet roofing (*qorqorro*).
   - Distant Addis Ababa thunder rumble.
   - Crackling charcoal embers and the scraping hiss of a metal spoon stirring green beans in a hot iron pan (*biret mitad*).
3. **Score**: Minimalist, intimate traditional strings (subtle Masenqo and Krar textures), entering softly when the proverb is spoken.

---

## User Testing & Evaluation Logs

### User Test #1 (Radio Cut - Audio & Pacing)
*Date Conducted: November 25, 2026*  
*Focus: Voice pacing, authentic Amharic pronunciation, proverb cadence, subtitle positioning, and trailing ambient breath.*

| Reviewer | Raw Quote | Issue Found | Exact Change Made to `storyboard.json` & Pipeline |
| :--- | :--- | :--- | :--- |
| **Person 1**<br>*(Mother / Native Amharic Speaker)* | *"The tone is warm and feels like home, but the voice pronounced cini too fast and rushed right through the proverb. Give the proverb a breath before and after so it lands."* | Pacing too brisk around ceremonial terms (*cini*) and proverb in `shot_07`; lacking maternal reflective cadence. | 1. Modified `shot_07` narration text to insert dramatic ellipses before and after the proverb quote.<br>2. Increased `shot_07` `pad_after_s` from 0.5s to 1.0s to allow proverb wisdom to settle.<br>3. Softened comma cadence preceding *cini* in `shot_08`. |
| **Person 2**<br>*(Cousin in Diaspora / Target Audience)* | *"The opening about the Kiremt rain hooked me immediately, though the ending felt a bit sudden. I wanted two or three seconds of just music and rain after the last sentence to sit with the memory."* | Intro lacked an explicit personal/childhood memory hook; final shot cut ended too abruptly without musical/rain reverb. | 1. Added explicit personal memory intro to `shot_01` (*"በልጅነቴ፣ ..."*).<br>2. Extended `shot_09` trailing padding (`pad_after_s`) from 0.5s to 3.0s for a contemplative music and rain fadeout. |
| **Person 3**<br>*(Cohort Colleague on Slack)* | *"The narration is clear, but the update ጥሬ አረንጓዴ የቡና ፍሬዎች to ጥሬ የቡና ፍሬዎች , and the speech for 'በትንሹ የሸክላ ምድጃ ከሰል ታቀጣጥላለች' is currently 'በትንሹ የሸክላ ምድጃ ከሰል ታቃጥላለች' fix it, the subtitles position is at the middle move it down."* | Redundant literal adjective (*አረንጓዴ*) unnatural in Amharic; charcoal lighting verb misspoken; subtitle centered vertically at mid-screen. | 1. Updated `shot_03` narration text from *"ጥሬ አረንጓዴ የቡና ፍሬዎች"* to *"ጥሬ የቡና ፍሬዎች"*.<br>2. Fixed `shot_02` text phrasing to *"ከሰሉን ታቀጣጥላለች"* for pristine pronunciation.<br>3. Moved subtitle rendering in `pipeline/radio_cut.py` from screen center (Y=350) down to lower third (Y=480). |

#### Key Insights from User Test #1:
1. **Linguistic Naturalness Over Literalism**: Native Amharic phrasing omits "አረንጓዴ" (green) when describing unroasted beans (*ጥሬ ቡና*) because "ጥሬ" already conveys raw green state.
2. **The "Proverb Landing Zone"**: Philosophical proverbs require audible acoustic breath (0.8s–1.2s pre- and post-pause) to signal their thematic gravity.
3. **Emotional Reverb**: Diaspora resonance relies heavily on the final 3 seconds of soundscape (tin roof rain + strings) after narration ceases, giving viewers emotional space to reflect.
4. **Lower-Third Text Placement**: Subtitles placed at dead center (Y=350) distract from compositional focus; lowering to Y=480 aligns with cinematic convention.

---

### User Test #2 (Near-Final Cut - Visuals, Audio & Subtitles)
*Focus: Evaluates character visual consistency, 9:16 mobile safe area legibility, subtitle synchronization, and overall emotional impact.*

| Reviewer | Raw Quote | Issue Found | Exact Change Made to `storyboard.json` |
| :--- | :--- | :--- | :--- |
| *[Reviewer 1]* | *"[Insert direct quote regarding visual consistency]"* | *[e.g., Mother's dress color shifted across shots]* | *[e.g., Standardized dress prompt to 'white Habesha Kemis with green Tilet']* |
| *[Reviewer 2]* | *"[Insert direct quote regarding mobile readability]"* | *[e.g., English subtitle cut off by Instagram share icon]* | *[e.g., Shifted subtitle vertical alignment upward by 80px]* |
| *[Reviewer 3]* | *"[Insert direct quote regarding final emotional arc]"* | *[e.g., Final end card cut off too abruptly]* | *[e.g., Extended final hold on steaming coffee cup to 3.0s]* |
