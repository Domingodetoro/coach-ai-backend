import express from 'express';
import cors from 'cors';
import Anthropic from '@anthropic-ai/sdk';
import OpenAI, { toFile } from 'openai';
import multer from 'multer';
import fs from 'fs';
import os from 'os';
import { createReadStream } from 'fs';

const app = express();
app.use(cors());
app.use(express.json({ limit: '2mb' }));

const anthropic = new Anthropic();
const upload    = multer({ dest: os.tmpdir() });

// System prompts cached across requests (prompt caching via cache_control)
const SYSTEM_INTERVIEW = [
  {
    type: 'text',
    text: `Eres un coach experto en entrevistas laborales con 15 años de experiencia en recursos humanos.
Evalúas respuestas de candidatos de forma objetiva, constructiva y motivadora.
Hablas en español de forma directa y sin rodeos. Eres específico, no genérico.
Responde SIEMPRE y ÚNICAMENTE en formato JSON válido, sin texto fuera del JSON.`,
    cache_control: { type: 'ephemeral' }
  }
];

const SYSTEM_CV = [
  {
    type: 'text',
    text: `Eres un experto en optimización de CVs para sistemas ATS y reclutadores de empresas top.
Analizas currículums y das feedback específico, accionable y medible.
Hablas en español. No eres genérico: si ves "Manejé campañas", dices exactamente cómo reescribirlo.
Responde SIEMPRE y ÚNICAMENTE en formato JSON válido, sin texto fuera del JSON.`,
    cache_control: { type: 'ephemeral' }
  }
];

function extractJSON(text) {
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) throw new Error('No JSON encontrado en respuesta');
  return JSON.parse(match[0]);
}

app.get('/health', (_, res) => res.json({ ok: true, version: '2.0' }));

app.post('/api/transcribe', upload.single('audio'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'Sin archivo de audio' });
  console.log('[transcribe] recibido:', req.file.size, 'bytes', req.file.mimetype);
  try {
    const openaiClient = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    const audioFile = await toFile(createReadStream(req.file.path), 'audio.m4a', { type: 'audio/m4a' });
    const transcription = await openaiClient.audio.transcriptions.create({
      file: audioFile,
      model: 'whisper-1',
      language: 'es',
    });
    console.log('[transcribe] resultado:', transcription.text);
    res.json({ transcript: transcription.text });
  } catch (e) {
    console.error('[transcribe] error:', e.message);
    res.status(500).json({ error: e.message });
  } finally {
    fs.unlink(req.file.path, () => {});
  }
});

app.post('/api/analyze-answer', async (req, res) => {
  const { role, question, answer } = req.body;
  if (!answer?.trim()) return res.json({ score: 0, tip: 'Respuesta vacía.', strengths: [], improvements: [] });

  try {
    const response = await anthropic.messages.create({
      model: 'claude-sonnet-4-6',
      max_tokens: 500,
      system: SYSTEM_INTERVIEW,
      messages: [{
        role: 'user',
        content: `Rol al que aplica: ${role}
Pregunta de entrevista: "${question}"
Respuesta del candidato: "${answer}"

Evalúa la respuesta y retorna este JSON exacto:
{
  "score": <número 0-100>,
  "strengths": [<máx 2 frases cortas de lo que hizo bien>],
  "improvements": [<máx 2 frases de cómo mejorar específicamente>],
  "tip": "<una acción concreta que puede hacer ahora mismo>"
}`
      }]
    });

    res.json(extractJSON(response.content[0].text));
  } catch (e) {
    console.error('[analyze-answer]', e.message);
    res.status(500).json({ error: e.message });
  }
});

app.post('/api/analyze-cv', async (req, res) => {
  const { cvText, targetRole } = req.body;
  if (!cvText?.trim()) return res.status(400).json({ error: 'CV vacío' });

  try {
    const response = await anthropic.messages.create({
      model: 'claude-sonnet-4-6',
      max_tokens: 1500,
      system: SYSTEM_CV,
      messages: [{
        role: 'user',
        content: `Rol objetivo: ${targetRole || 'General'}

Texto del CV:
${cvText.slice(0, 4000)}

Analiza el CV y retorna este JSON exacto (5-6 issues, mezcla de críticos, mejorables y buenos):
{
  "score": <número 0-100>,
  "issues": [
    {
      "severity": "<critical|warning|good>",
      "title": "<título corto del hallazgo>",
      "body": "<explicación específica con ejemplo concreto de cómo mejorarlo>"
    }
  ],
  "keywords_missing": [<palabras clave del rol que faltan en el CV>],
  "keywords_present": [<palabras clave del rol que ya están>],
  "summary": "<2 frases: diagnóstico general y principal prioridad>"
}`
      }]
    });

    res.json(extractJSON(response.content[0].text));
  } catch (e) {
    console.error('[analyze-cv]', e.message);
    res.status(500).json({ error: e.message });
  }
});

app.post('/api/generate-report', async (req, res) => {
  const { role, answers } = req.body;
  if (!answers?.length) return res.status(400).json({ error: 'Sin respuestas' });

  const answersText = answers
    .map((a, i) =>
      `P${i + 1}: "${a.question}"\nRespuesta: "${(a.answer || '(sin respuesta)').slice(0, 500)}" | ${a.wordCount} palabras | ${a.fillerCount} muletillas | ${a.duration}s`
    )
    .join('\n\n');

  try {
    const response = await anthropic.messages.create({
      model: 'claude-sonnet-4-6',
      max_tokens: 900,
      system: SYSTEM_INTERVIEW,
      messages: [{
        role: 'user',
        content: `Rol: ${role}
Entrevista completa:
${answersText}

Genera el reporte final en este JSON exacto:
{
  "overall_score": <número 0-100>,
  "tag": "<etiqueta como 'Excelente desempeño' o 'En desarrollo'>",
  "summary": "<2 oraciones: diagnóstico general honesto y posicionamiento vs otros candidatos>",
  "scores": {
    "claridad": <0-100>,
    "confianza_vocal": <0-100>,
    "lenguaje_corporal": <0-100>,
    "contenido": <0-100>,
    "ritmo": <0-100>
  },
  "strengths": "<párrafo de 2-3 oraciones sobre lo que hizo bien, con ejemplos de sus respuestas>",
  "improvements": "<párrafo de 2-3 oraciones con consejos específicos y accionables>",
  "next_step": "<una sola acción concreta que debe hacer antes de su próxima entrevista real>"
}`
      }]
    });

    res.json(extractJSON(response.content[0].text));
  } catch (e) {
    console.error('[generate-report]', e.message);
    res.status(500).json({ error: e.message });
  }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`\n✅ Coach AI Server corriendo → http://localhost:${PORT}`);
  console.log(`   API Key: ${process.env.ANTHROPIC_API_KEY ? '✓ configurada' : '✗ falta ANTHROPIC_API_KEY'}\n`);
});
