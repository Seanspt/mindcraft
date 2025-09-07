import os from 'os';
import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import FormData from 'form-data';
import fetch from 'node-fetch';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// --- Argument Parsing ---
const args = {};
for (let i = 2; i < process.argv.length; i++) {
    const arg = process.argv[i];
    if (arg.startsWith('--')) {
        const key = arg.substring(2);
        if (i + 1 < process.argv.length && !process.argv[i + 1].startsWith('--')) {
            args[key] = process.argv[i + 1];
            i++;
        } else {
            args[key] = true;
        }
    }
}

const DEFAULTS = {
    host: '192.168.3.78',
    port: 50000,
    mode: 'instruct2',
    tts_text: '',
    spk_id: '中文女',
    prompt_text: '用细竹扎出骨架，在香素纸上绘制心仪的图案，糊上纸系好线',
    prompt_wav: path.resolve(__dirname, '../../assets/刻晴.wav'),
    instruct_text: '模仿小猪佩奇的风格快速说',
    tts_wav: 'demo_nodejs.wav'
};

const config = { ...DEFAULTS, ...args };

function createWavHeader(sampleRate, numChannels, bitsPerSample, dataLength) {
    const buffer = Buffer.alloc(44);

    buffer.write('RIFF', 0);
    buffer.writeUInt32LE(36 + dataLength, 4);
    buffer.write('WAVE', 8);
    buffer.write('fmt ', 12);
    buffer.writeUInt32LE(16, 16);
    buffer.writeUInt16LE(1, 20);
    buffer.writeUInt16LE(numChannels, 22);
    buffer.writeUInt32LE(sampleRate, 24);
    buffer.writeUInt32LE(sampleRate * numChannels * (bitsPerSample / 8), 28);
    buffer.writeUInt16LE(numChannels * (bitsPerSample / 8), 32);
    buffer.writeUInt16LE(bitsPerSample, 34);
    buffer.write('data', 36);
    buffer.writeUInt32LE(dataLength, 40);

    return buffer;
}

async function main() {
    const url = `http://${config.host}:${config.port}/inference_${config.mode}`;
    console.log(`[INFO] Target URL: ${url}`);

    let response;

    try {
        const formData = new FormData();
        console.log("[INFO] Assembling form data for mode:", config.mode);
        switch (config.mode) {
            case 'sft':
                formData.append('tts_text', config.tts_text);
                formData.append('spk_id', config.spk_id);
                break;
            case 'zero_shot':
                formData.append('tts_text', config.tts_text);
                formData.append('prompt_text', config.prompt_text);
                formData.append('prompt_wav', fs.readFileSync(config.prompt_wav), { filename: path.basename(config.prompt_wav) });
                break;
            case 'cross_lingual':
                formData.append('tts_text', config.tts_text);
                formData.append('prompt_wav', fs.readFileSync(config.prompt_wav), { filename: path.basename(config.prompt_wav) });
                break;
            case 'instruct':
                formData.append('tts_text', config.tts_text);
                formData.append('spk_id', config.spk_id);
                formData.append('instruct_text', config.instruct_text);
                break;
            case 'instruct2':
                formData.append('tts_text', config.tts_text);
                formData.append('instruct_text', config.instruct_text);
                formData.append('prompt_wav', fs.readFileSync(config.prompt_wav), { filename: path.basename(config.prompt_wav) });
                break;
            default:
                throw new Error(`Unknown mode: ${config.mode}`);
        }
        console.log("[INFO] Form data assembled.");

        const headers = formData.getHeaders();
        console.log("[INFO] Sending request to server... (If the script hangs here, the issue is likely with the network connection or the server not accepting the request).")
        
        response = await fetch(url, { method: 'POST', body: formData, headers: headers });
        
        console.log(`[INFO] Response received. Status: ${response.status}`);

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server responded with status ${response.status}: ${errorText}`);
        }

        console.log("[INFO] Reading response body stream... (If it hangs here, the server accepted the request but is not sending back data). ");
        const audioChunks = [];
        for await (const chunk of response.body) {
            audioChunks.push(chunk);
            console.log(`  - Received chunk of size: ${chunk.length}`);
        }
        console.log("[INFO] Finished reading response stream.");

        const pcmData = Buffer.concat(audioChunks);
        const wavHeader = createWavHeader(22050, 1, 16, pcmData.length);
        const wavData = Buffer.concat([wavHeader, pcmData]);
        console.log(`[INFO] WAV data prepared. Total size: ${wavData.length} bytes.`);

        // Play audio directly
        if (process.platform !== 'darwin') {
            console.warn('Audio playback is only supported on macOS (afplay). Saving to file instead.');
            fs.writeFileSync(config.tts_wav, wavData);
            console.log(`[INFO] Successfully saved audio to ${config.tts_wav}`);
            return;
        }

        const tempFilePath = path.join(os.tmpdir(), `cosyvoice-playback-${Date.now()}.wav`);
        
        try {
            fs.writeFileSync(tempFilePath, wavData);
            console.log(`[INFO] Starting playback from temporary file: ${tempFilePath}`);
            const player = spawn('afplay', [tempFilePath]);

            let errorOutput = '';
            player.stderr.on('data', (data) => {
                errorOutput += data.toString();
            });

            player.on('close', (code) => {
                if (code === 0) {
                    console.log('[INFO] Playback finished.');
                } else {
                    console.error(`[ERROR] Playback process exited with code ${code}.`);
                    if (errorOutput) {
                        console.error('[afplay stderr]:', errorOutput);
                    }
                }
                // Always clean up the temporary file
                fs.unlinkSync(tempFilePath);
                console.log(`[INFO] Deleted temporary file.`);
            });

            player.on('error', (err) => {
                console.error('[ERROR] Failed to start playback process. Is afplay in your PATH?', err);
                // Clean up if player fails to start
                fs.unlinkSync(tempFilePath);
            });

        } catch (writeErr) {
            console.error(`[ERROR] Failed to write temporary audio file: ${writeErr.message}`);
        }

    } catch (error) {
        console.error("\n--- SCRIPT FAILED ---");
        console.error("Message:", error.message);
        console.error("Stack:", error.stack);
        console.error("---------------------");
    }
}

main();
