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
*Focus: Evaluates voice pacing, pronunciation of the Amharic proverb, Foley balance, and emotional resonance before generating video shots.*

| Reviewer | Raw Quote | Issue Found | Exact Change Made to `storyboard.json` |
| :--- | :--- | :--- | :--- |
| *[Reviewer 1]* | *"[Insert direct quote from diaspora listener]"* | *[e.g., Narration too fast during proverb delivery]* | *[e.g., Increased shot_03 duration from 3.2s to 4.5s]* |
| *[Reviewer 2]* | *"[Insert direct quote from native Amharic speaker]"* | *[e.g., Rain Foley overpowering the voice]* | *[e.g., Reduced background rain audio volume to -18dB]* |
| *[Reviewer 3]* | *"[Insert direct quote from mobile listener]"* | *[e.g., Unclear transition between intro and roasting]* | *[e.g., Inserted 0.8s pause before proverb narration]* |

---

### User Test #2 (Near-Final Cut - Visuals, Audio & Subtitles)
*Focus: Evaluates character visual consistency, 9:16 mobile safe area legibility, subtitle synchronization, and overall emotional impact.*

| Reviewer | Raw Quote | Issue Found | Exact Change Made to `storyboard.json` |
| :--- | :--- | :--- | :--- |
| *[Reviewer 1]* | *"[Insert direct quote regarding visual consistency]"* | *[e.g., Mother's dress color shifted across shots]* | *[e.g., Standardized dress prompt to 'white Habesha Kemis with green Tilet']* |
| *[Reviewer 2]* | *"[Insert direct quote regarding mobile readability]"* | *[e.g., English subtitle cut off by Instagram share icon]* | *[e.g., Shifted subtitle vertical alignment upward by 80px]* |
| *[Reviewer 3]* | *"[Insert direct quote regarding final emotional arc]"* | *[e.g., Final end card cut off too abruptly]* | *[e.g., Extended final hold on steaming coffee cup to 3.0s]* |
