import React, { useState, useEffect, useRef } from 'react';
import { AgentConfig, UserProfile, ActiveGenerationTask } from '../types';
import { generateWanVideo, generateLtxVideo, proxyFetch, DEFAULT_LTX_VIDEO_SPACE_URL } from '../services/huggingFaceService';
import { generateVideoPromptWithAI, generateVideoCommentaryWithAI } from '../services/geminiService';
import { DEFAULT_WAN_VIDEO_SPACE_URL } from '../services/dbService';
import { downloadMediaFile } from '../services/exportHelper';

interface ChatHistoryImage {
  url: string;
  prompt?: string;
}

interface WanVideoModalProps {
  isOpen: boolean;
  onClose: () => void;
  initialStartImage?: string | null;
  initialPrompt?: string;
  userProfile?: UserProfile;
  agentConfig: AgentConfig;
  themeHex?: string;
  isBackgroundDark?: boolean;
  chatHistoryImages?: ChatHistoryImage[];
  onVideoGenerated?: (videoUrl: string, prompt: string, startImage: string, duration?: number, commentary?: string) => Promise<void> | void;
  onStartGeneration?: (task: ActiveGenerationTask) => void;
  onUpdateGeneration?: (agentId: string, statusText: string, type?: 'text' | 'pap' | 'audio') => void;
  onFinishGeneration?: (agentId: string, type?: 'text' | 'pap' | 'audio') => void;
}

export const WanVideoModal: React.FC<WanVideoModalProps> = ({
  isOpen,
  onClose,
  initialStartImage = null,
  initialPrompt = '',
  userProfile,
  agentConfig,
  themeHex = '#ec4899',
  isBackgroundDark = true,
  chatHistoryImages = [],
  onVideoGenerated,
  onStartGeneration,
  onFinishGeneration
}) => {
  const [startImage, setStartImage] = useState<string | null>(initialStartImage);
  const [endImage, setEndImage] = useState<string | null>(null);
  const [userIdea, setUserIdea] = useState<string>('');
  const [prompt, setPrompt] = useState<string>('');
  const [duration, setDuration] = useState<number>(5.0);
  const [isEnhancingPrompt, setIsEnhancingPrompt] = useState<boolean>(false);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [generationStatus, setGenerationStatus] = useState<string>('');
  const [generatedVideoUrl, setGeneratedVideoUrl] = useState<string | null>(null);
  const [analyzedCommentary, setAnalyzedCommentary] = useState<string | null>(null);
  const [isSavingToChat, setIsSavingToChat] = useState<boolean>(false);
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [identityImage, setIdentityImage] = useState<string | null>(null);
  const [identityStrength, setIdentityStrength] = useState<number>(0.45);
  const [isDraggingStart, setIsDraggingStart] = useState<boolean>(false);
  const [isDraggingEnd, setIsDraggingEnd] = useState<boolean>(false);
  const [isDraggingIdentity, setIsDraggingIdentity] = useState<boolean>(false);

  // LTX 2.3 Audio Settings
  const [audioRefEnabled, setAudioRefEnabled] = useState(false);
  const [audioRefGuidanceScale, setAudioRefGuidanceScale] = useState(3.0);
  const [audioRefStemSep, setAudioRefStemSep] = useState(false);
  const [audioRefNormalize, setAudioRefNormalize] = useState(true);
  const [audioRefFile, setAudioRefFile] = useState<string | null>(null);
  const audioFileInputRef = useRef<HTMLInputElement>(null);

  const startFileInputRef = useRef<HTMLInputElement>(null);
  const endFileInputRef = useRef<HTMLInputElement>(null);
  const identityFileInputRef = useRef<HTMLInputElement>(null);
  const timerRef = useRef<any>(null);
  const videoPlayerRef = useRef<HTMLVideoElement>(null);

  // Sync initial state when modal opens
  useEffect(() => {
    if (isOpen) {
      if (initialStartImage) {
        setStartImage(initialStartImage);
      }
      if (initialPrompt) {
        setPrompt(initialPrompt);
        setUserIdea(initialPrompt);
      } else if (!prompt) {
        // Default default motion prompt
        setPrompt("cinematic motion, subtle gentle smile looking at camera, soft breathing motion, camera slow smooth push in, soft natural lighting, photorealistic, 4k");
      }
      setErrorMessage(null);
      setAnalyzedCommentary(null);
      setIsSavingToChat(false);
    }
  }, [isOpen, initialStartImage, initialPrompt]);

  // Elapsed time tracker during generation
  useEffect(() => {
    if (isGenerating) {
      setElapsedSeconds(0);
      timerRef.current = setInterval(() => {
        setElapsedSeconds(prev => prev + 1);
      }, 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isGenerating]);

  if (!isOpen) return null;

  // File Upload Handlers
  const handleStartFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const result = event.target?.result as string;
      if (result) setStartImage(result);
    };
    reader.readAsDataURL(file);
    e.target.value = '';
  };

  const handleEndFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const result = event.target?.result as string;
      if (result) setEndImage(result);
    };
    reader.readAsDataURL(file);
    e.target.value = '';
  };

  const handleIdentityFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const result = event.target?.result as string;
      if (result) setIdentityImage(result);
    };
    reader.readAsDataURL(file);
    e.target.value = '';
  };

  // AI Prompt Engineering Action
  const handleEnhancePromptWithAI = async () => {
    if (isEnhancingPrompt) return;
    setIsEnhancingPrompt(true);
    setErrorMessage(null);

    try {
      const characterContext = `${agentConfig.name}, ${agentConfig.currentOutfit || ''}, ${agentConfig.personality || ''}`;
      const requestText = userIdea.trim() || prompt.trim() || "animate character with cinematic motion";
      const enhanced = await generateVideoPromptWithAI(requestText, characterContext, userProfile);
      setPrompt(enhanced);
    } catch (err: any) {
      console.warn("[WanVideoModal] Gagal enhance prompt:", err);
    } finally {
      setIsEnhancingPrompt(false);
    }
  };

  const handleDrop = (e: React.DragEvent, setter: (val: string | null) => void) => {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const result = event.target?.result as string;
        if (result) setter(result);
      };
      reader.readAsDataURL(file);
    }
  };

  // Generate Video Action
  const handleGenerateVideo = async () => {
    if (!startImage) {
      setErrorMessage("Silakan tentukan Start Frame (gambar awal) terlebih dahulu.");
      return;
    }

    setIsGenerating(true);
    setErrorMessage(null);
    const model = userProfile?.videoGeneratorModel || 'wan2.2';
    setGenerationStatus(`Menghubungkan ke ${model === 'ltx2.3' ? 'LTX 2.3' : 'Wan 2.2'} Studio...`);

    try {
      let finalPrompt = prompt.trim();
      if (!finalPrompt && userIdea.trim()) {
        setGenerationStatus("Gemini 3.1 Flash Lite sedang memformulasikan prompt sinematik...");
        const characterContext = `${agentConfig.name}, ${agentConfig.currentOutfit || ''}, ${agentConfig.personality || ''}`;
        finalPrompt = await generateVideoPromptWithAI(userIdea.trim(), characterContext, userProfile, model, duration);
        setPrompt(finalPrompt);
      }
      if (!finalPrompt) {
        finalPrompt = model === 'ltx2.3' 
          ? "cinematic motion, smooth camera push-in, natural motion, 4k" 
          : "cinematic motion, smooth camera push-in, natural motion, 4k";
      }

      let videoResult;
      if (model === 'ltx2.3') {
        videoResult = await generateLtxVideo({
          startImage,
          prompt: finalPrompt,
          duration: duration,
          spaceUrl: userProfile?.hfVideoSpaceUrl || DEFAULT_LTX_VIDEO_SPACE_URL,
          tokens: userProfile?.hfTokens || '',
          audioRefFile: audioRefFile,
          audioRefEnabled,
          audioRefGuidanceScale,
          audioRefStemSep,
          audioRefNormalize,
          onStatusUpdate: (status) => setGenerationStatus(status)
        });
      } else {
        videoResult = await generateWanVideo({
          startImage,
          endImage: endImage || null,
          prompt: finalPrompt,
          duration: duration,
          fps: 64,
          spaceUrl: userProfile?.hfVideoSpaceUrl || DEFAULT_WAN_VIDEO_SPACE_URL,
          tokens: userProfile?.hfTokens || '',
          identityImage,
          identityStrength,
          onStatusUpdate: (status) => setGenerationStatus(status)
        });
      }

      setGeneratedVideoUrl(videoResult);

      // Konversi video ke base64... (sama seperti sebelumnya)
      let analysisVideoUrl = videoResult;
      if (!analysisVideoUrl.startsWith('data:')) {
        try {
          setGenerationStatus("Mengonversi video ke memori lokal (base64) untuk analisis AI...");
          const resp = await proxyFetch(analysisVideoUrl);
          if (resp.ok) {
            const blob = await resp.blob();
            analysisVideoUrl = await new Promise<string>((resolve, reject) => {
              const reader = new FileReader();
              reader.onload = () => resolve(reader.result as string);
              reader.onerror = reject;
              reader.readAsDataURL(blob);
            });
            setGeneratedVideoUrl(analysisVideoUrl);
          }
        } catch (e) {
          console.warn("[WanVideoModal] Pre-convert video to base64 failed, using URL:", e);
        }
      }

      // Mulai analisa video... (sama seperti sebelumnya)
      generateVideoCommentaryWithAI({
        videoUrl: analysisVideoUrl,
        videoPrompt: finalPrompt,
        startImage,
        agentConfig,
        userProfile,
        duration
      }).then((comm) => {
        setAnalyzedCommentary(comm);
      }).catch((err) => {
        console.warn("[WanVideoModal] Pre-analyze commentary error:", err);
      });
    } catch (err: any) {
      console.error("[WanVideoModal] Generate Video Error:", err);
      setErrorMessage(err?.message || "Gagal menghasilkan video.");
    } finally {
      setIsGenerating(false);
    }
  };

  // Download Generated Video
  const handleDownload = () => {
    if (!generatedVideoUrl) return;
    const now = new Date();
    const dateStr = `${now.getDate().toString().padStart(2, '0')}-${(now.getMonth() + 1).toString().padStart(2, '0')}-${now.getFullYear()}`;
    const timeStr = `${now.getHours().toString().padStart(2, '0')}-${now.getMinutes().toString().padStart(2, '0')}-${now.getSeconds().toString().padStart(2, '0')}`;
    const fileName = `story_${agentConfig.name.toLowerCase().replace(/\s+/g, '_')}_${dateStr}_${timeStr}.mp4`;
    downloadMediaFile(generatedVideoUrl, fileName, 'video/mp4');
  };

  // Save/Insert to Chat
  const handleInsertToChat = async () => {
    if (!generatedVideoUrl || !startImage || isSavingToChat) return;
    setIsSavingToChat(true);
    try {
      let finalVideoUrl = generatedVideoUrl;
      // Pastikan video sudah dalam base64 lokal saat disimpan ke chat
      if (!finalVideoUrl.startsWith('data:')) {
        try {
          setGenerationStatus("Menyimpan video ke memori lokal dalam base64...");
          const resp = await proxyFetch(finalVideoUrl);
          if (resp.ok) {
            const blob = await resp.blob();
            finalVideoUrl = await new Promise<string>((resolve, reject) => {
              const reader = new FileReader();
              reader.onload = () => resolve(reader.result as string);
              reader.onerror = reject;
              reader.readAsDataURL(blob);
            });
          }
        } catch (convErr) {
          console.warn("[WanVideoModal] Failed to convert video to base64 on save, using URL:", convErr);
        }
      }

      let commentary = analyzedCommentary;
      if (!commentary) {
        setGenerationStatus("Menganalisa video untuk pesan chat...");
        const capturedAgentId = agentConfig.id || 'default';
        onStartGeneration?.({
          id: `vid-comment-${Date.now()}`,
          agentId: capturedAgentId,
          agentName: agentConfig.name,
          agentPic: agentConfig.profilePic,
          type: 'text',
          statusText: 'Menganalisa video...',
          startedAt: Date.now()
        });
        
        try {
          commentary = await generateVideoCommentaryWithAI({
            videoUrl: finalVideoUrl,
            videoPrompt: prompt.trim() || userIdea.trim(),
            startImage,
            agentConfig,
            userProfile,
            duration
          });
        } finally {
          onFinishGeneration?.(capturedAgentId, 'text');
        }
        
        setAnalyzedCommentary(commentary);
      }
      if (onVideoGenerated) {
        await onVideoGenerated(finalVideoUrl, prompt, startImage, duration, commentary);
      }
      onClose();
    } catch (err) {
      console.warn("[WanVideoModal] Error in handleInsertToChat:", err);
      if (onVideoGenerated) {
        await onVideoGenerated(generatedVideoUrl, prompt, startImage, duration);
      }
      onClose();
    } finally {
      setIsSavingToChat(false);
    }
  };

  return (
    <div 
      className="fixed inset-0 z-[250] flex items-center justify-center p-3 sm:p-5 bg-black/85 backdrop-blur-2xl animate-in fade-in duration-300"
      onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
      onDrop={(e) => { e.preventDefault(); e.stopPropagation(); }}
    >
      <div 
        className={`relative w-full max-w-4xl max-h-[92vh] overflow-y-auto rounded-[32px] border border-white/15 shadow-[0_25px_70px_rgba(0,0,0,0.8)] flex flex-col ${
          isBackgroundDark ? 'bg-zinc-950/95 text-white' : 'bg-neutral-900/95 text-white'
        } custom-scrollbar`}
        style={{
          boxShadow: `0 0 80px ${themeHex}20`
        }}
      >
        {/* Hidden File Inputs */}
        <input 
          type="file" 
          ref={startFileInputRef} 
          className="hidden" 
          accept="image/*" 
          onChange={handleStartFileChange} 
        />
        <input 
          type="file" 
          ref={endFileInputRef} 
          className="hidden" 
          accept="image/*" 
          onChange={handleEndFileChange} 
        />

        {/* Modal Header */}
        <div className="sticky top-0 z-20 flex items-center justify-between px-6 py-4 bg-zinc-950/80 backdrop-blur-xl border-b border-white/10 select-none">
          <div className="flex items-center gap-3">
            <div 
              className="w-10 h-10 rounded-2xl flex items-center justify-center shadow-lg border border-white/20"
              style={{ backgroundColor: `${themeHex}30`, color: themeHex }}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 24 24" fill="currentColor">
                <path d="M4 4.5A2.5 2.5 0 0 1 6.5 2H18a2.5 2.5 0 0 1 2.5 2.5v14.25a.75.75 0 0 1-1.28.53L16 16H6.5A2.5 2.5 0 0 1 4 13.5V4.5Z" />
              </svg>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base sm:text-lg font-black tracking-tight text-white flex items-center gap-1.5">
                  Video Studio
                </h2>
              </div>
              <p className="text-[11px] text-white/50 font-medium">
                Image-to-Video Engine
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            disabled={isGenerating}
            className="p-2.5 rounded-full bg-white/5 hover:bg-white/15 text-white/60 hover:text-white transition-all active:scale-95 disabled:opacity-30 cursor-pointer"
            title="Tutup Modal"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-5 sm:p-7 space-y-6 flex-1">
          {/* Error Banner */}
          {errorMessage && (
            <div className="p-3.5 rounded-2xl bg-red-500/15 border border-red-500/40 text-red-200 text-xs flex items-start gap-2.5 animate-in fade-in duration-200">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-red-400 mt-0.5 shrink-0" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <div className="flex-1 leading-relaxed">
                <span className="font-bold">Terjadi Kesalahan:</span> {errorMessage}
              </div>
            </div>
          )}

          {/* Section 1: Frame Slots (Start Frame & Optional End Frame) */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Slot 1: Start Frame */}
            <div 
              className={`flex flex-col gap-2 p-4 rounded-2xl bg-white/5 border-2 transition-all duration-300 relative ${isDraggingStart ? 'border-indigo-500 scale-[1.02] shadow-2xl bg-indigo-500/10' : 'border-white/10'}`}
              onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); setIsDraggingStart(true); setIsDraggingEnd(false); setIsDraggingIdentity(false); }}
              onDragLeave={(e) => { e.preventDefault(); e.stopPropagation(); setIsDraggingStart(false); }}
              onDrop={(e) => { 
                e.preventDefault(); 
                e.stopPropagation(); 
                setIsDraggingStart(false); 
                const file = e.dataTransfer.files?.[0];
                if (file && file.type.startsWith('image/')) {
                  const reader = new FileReader();
                  reader.onload = (event) => {
                    const result = event.target?.result as string;
                    if (result) setStartImage(result);
                  };
                  reader.readAsDataURL(file);
                }
              }}
            >
              {isDraggingStart && (
                <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-indigo-500/25 backdrop-blur-sm rounded-2xl animate-in fade-in duration-200 pointer-events-none">
                  <div className="w-12 h-12 rounded-full bg-indigo-500 flex items-center justify-center shadow-xl mb-2 animate-bounce">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a2 2 0 002 2h12a2 2 0 002-2v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                  </div>
                  <span className="text-white text-[9px] font-black uppercase tracking-[0.2em] drop-shadow-md">Lepas untuk Start Frame</span>
                </div>
              )}

              <div className="flex items-center justify-between">
                <span className="text-xs font-black uppercase tracking-wider text-white/90 flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
                  Start Frame (Wajib)
                </span>
                <div className="flex gap-2">
                  {startImage && (
                    <button type="button" onClick={() => setStartImage(null)} disabled={isGenerating} className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-lg bg-red-500/20 text-red-300 hover:bg-red-500/30 transition-all cursor-pointer">Hapus</button>
                  )}
                  <button
                    type="button"
                    onClick={() => startFileInputRef.current?.click()}
                    disabled={isGenerating}
                    className="text-[10px] font-bold uppercase tracking-wider px-2.5 py-1 rounded-xl bg-white/10 hover:bg-white/20 text-white transition-all cursor-pointer"
                  >
                    {startImage ? 'Ganti' : 'Unggah'}
                  </button>
                </div>
              </div>

              {startImage ? (
                <div className="relative aspect-[4/3] rounded-xl overflow-hidden bg-black/40 border border-white/10 group">
                  <img src={startImage} alt="Start Frame" className="w-full h-full object-contain" />
                </div>
              ) : (
                <div 
                  onClick={() => startFileInputRef.current?.click()}
                  className="aspect-[4/3] rounded-xl border-2 border-dashed border-white/20 hover:border-white/40 bg-black/20 flex flex-col items-center justify-center gap-2 cursor-pointer transition-all hover:bg-white/5"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-white/40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2-2v12a2 2 0 002 2z" />
                  </svg>
                  <span className="text-xs font-bold text-white/60">Klik atau Drag & Drop Start Frame</span>
                  <span className="text-[10px] text-white/30">PNG / JPG / WEBP</span>
                </div>
              )}

              {/* Chat PAP quick selection carousel */}
              {chatHistoryImages.length > 0 && (
                <div className="mt-1 space-y-1">
                  <span className="text-[9px] font-bold text-white/40 uppercase tracking-wider">
                    Pilih dari Foto Chat Terkini:
                  </span>
                  <div className="flex gap-1.5 overflow-x-auto pb-1 custom-scrollbar">
                    {chatHistoryImages.slice(-8).map((img, idx) => (
                      <button
                        key={idx}
                        type="button"
                        onClick={() => setStartImage(img.url)}
                        className={`w-11 h-11 rounded-lg overflow-hidden border shrink-0 transition-all ${
                          startImage === img.url ? 'border-emerald-400 scale-105 ring-2 ring-emerald-400/30' : 'border-white/20 opacity-70 hover:opacity-100'
                        }`}
                        title="Gunakan foto ini sebagai Start Frame"
                      >
                        <img src={img.url} className="w-full h-full object-cover" alt="History" />
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Slot 2: End Frame (Optional) */}
            <div 
              className={`flex flex-col gap-2 p-4 rounded-2xl bg-white/5 border-2 transition-all duration-300 relative ${isDraggingEnd ? 'border-indigo-500 scale-[1.02] shadow-2xl bg-indigo-500/10' : 'border-white/10'}`}
              onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); setIsDraggingEnd(true); setIsDraggingStart(false); setIsDraggingIdentity(false); }}
              onDragLeave={(e) => { e.preventDefault(); e.stopPropagation(); setIsDraggingEnd(false); }}
              onDrop={(e) => { 
                e.preventDefault(); 
                e.stopPropagation(); 
                setIsDraggingEnd(false); 
                const file = e.dataTransfer.files?.[0];
                if (file && file.type.startsWith('image/')) {
                  const reader = new FileReader();
                  reader.onload = (event) => {
                    const result = event.target?.result as string;
                    if (result) setEndImage(result);
                  };
                  reader.readAsDataURL(file);
                }
              }}
            >
              {isDraggingEnd && (
                <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-indigo-500/25 backdrop-blur-sm rounded-2xl animate-in fade-in duration-200 pointer-events-none">
                  <div className="w-12 h-12 rounded-full bg-indigo-500 flex items-center justify-center shadow-xl mb-2 animate-bounce">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a2 2 0 002 2h12a2 2 0 002-2v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                  </div>
                  <span className="text-white text-[9px] font-black uppercase tracking-[0.2em] drop-shadow-md">Lepas untuk End Frame</span>
                </div>
              )}

              <div className="flex items-center justify-between">
                <span className="text-xs font-black uppercase tracking-wider text-white/70 flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-violet-400"></span>
                  End Frame (Opsional)
                </span>
                <div className="flex gap-2">
                  {endImage && (
                    <button type="button" onClick={() => setEndImage(null)} disabled={isGenerating} className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-lg bg-red-500/20 text-red-300 hover:bg-red-500/30 transition-all cursor-pointer">Hapus</button>
                  )}
                  <button
                    type="button"
                    onClick={() => endFileInputRef.current?.click()}
                    disabled={isGenerating}
                    className="text-[10px] font-bold uppercase tracking-wider px-2.5 py-1 rounded-xl bg-white/10 hover:bg-white/20 text-white transition-all cursor-pointer"
                  >
                    {endImage ? 'Ganti' : 'Unggah'}
                  </button>
                </div>
              </div>

              {endImage ? (
                <div className="relative aspect-[4/3] rounded-xl overflow-hidden bg-black/40 border border-white/10 group">
                  <img src={endImage} alt="End Frame" className="w-full h-full object-contain" />
                </div>
              ) : (
                <div 
                  onClick={() => endFileInputRef.current?.click()}
                  className="aspect-[4/3] rounded-xl border-2 border-dashed border-white/10 hover:border-white/30 bg-black/10 flex flex-col items-center justify-center gap-2 cursor-pointer transition-all hover:bg-white/5"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-7 w-7 text-white/30" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                  </svg>
                  <span className="text-xs font-semibold text-white/50">Tambah End Frame atau Drag & Drop</span>
                  <span className="text-[10px] text-white/30 text-center px-4">
                    Mengarahkan transisi gerakan menuju pose/posisi akhir
                  </span>
                </div>
              )}

              {/* Chat PAP quick selection carousel for End Frame */}
              {chatHistoryImages.length > 0 && (
                <div className="mt-1 space-y-1">
                  <span className="text-[9px] font-bold text-white/40 uppercase tracking-wider">
                    Pilih Frame Akhir dari Obrolan:
                  </span>
                  <div className="flex gap-1.5 overflow-x-auto pb-1 custom-scrollbar">
                    {chatHistoryImages.slice(-8).map((img, idx) => (
                      <button
                        key={idx}
                        type="button"
                        onClick={() => setEndImage(img.url)}
                        className={`w-11 h-11 rounded-lg overflow-hidden border shrink-0 transition-all ${
                          endImage === img.url ? 'border-violet-400 scale-105 ring-2 ring-violet-400/30' : 'border-white/20 opacity-70 hover:opacity-100'
                        }`}
                        title="Gunakan foto ini sebagai End Frame"
                      >
                        <img src={img.url} className="w-full h-full object-cover" alt="History" />
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* New Section: Identity Lock */}
          <div 
            className={`flex flex-col gap-2 p-4 rounded-2xl bg-white/5 border-2 transition-all duration-300 relative ${isDraggingIdentity ? 'border-indigo-500 scale-[1.02] shadow-2xl bg-indigo-500/10' : 'border-white/10'}`}
            onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); setIsDraggingIdentity(true); setIsDraggingStart(false); setIsDraggingEnd(false); }}
            onDragLeave={(e) => { e.preventDefault(); e.stopPropagation(); setIsDraggingIdentity(false); }}
            onDrop={(e) => { 
              e.preventDefault(); 
              e.stopPropagation(); 
              setIsDraggingIdentity(false); 
              const file = e.dataTransfer.files?.[0];
              if (file && file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = (event) => {
                  const result = event.target?.result as string;
                  if (result) setIdentityImage(result);
                };
                reader.readAsDataURL(file);
              }
            }}
          >
            {isDraggingIdentity && (
              <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-indigo-500/25 backdrop-blur-sm rounded-2xl animate-in fade-in duration-200 pointer-events-none">
                <div className="w-12 h-12 rounded-full bg-indigo-500 flex items-center justify-center shadow-xl mb-2 animate-bounce">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a2 2 0 002 2h12a2 2 0 002-2v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                </div>
                <span className="text-white text-[9px] font-black uppercase tracking-[0.2em] drop-shadow-md">Lepas untuk Identity Lock</span>
              </div>
            )}

            <div className="flex items-center justify-between">
              <span className="text-xs font-black uppercase tracking-wider text-white/90 flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-amber-400"></span>
                Identity Lock (Opsional) - Drag & Drop atau Unggah
              </span>
              <div className="flex gap-2">
                {identityImage && (
                  <button type="button" onClick={() => setIdentityImage(null)} disabled={isGenerating} className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-lg bg-red-500/20 text-red-300 hover:bg-red-500/30 transition-all cursor-pointer">Hapus</button>
                )}
                <button
                  type="button"
                  onClick={() => identityFileInputRef.current?.click()}
                  disabled={isGenerating}
                  className="text-[10px] font-bold uppercase tracking-wider px-2.5 py-1 rounded-xl bg-white/10 hover:bg-white/20 text-white transition-all cursor-pointer"
                >
                  {identityImage ? 'Ganti' : 'Unggah'}
                </button>
              </div>
            </div>
            
            <input 
              type="file" 
              ref={identityFileInputRef} 
              className="hidden" 
              accept="image/*" 
              onChange={handleIdentityFileChange} 
            />

            {identityImage && (
              <div className="mt-2 relative w-20 h-20 rounded-xl overflow-hidden bg-black/40 border border-white/10">
                <img src={identityImage} alt="Identity Lock" className="w-full h-full object-cover" />
              </div>
            )}

            <div className="mt-2">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-bold text-white/70">At most this strong</span>
                <span className="text-sm font-black text-amber-400">{identityStrength.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={identityStrength}
                onChange={(e) => setIdentityStrength(parseFloat(e.target.value))}
                disabled={isGenerating || !identityImage}
                className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer accent-amber-400 disabled:opacity-40"
              />
            </div>
          </div>

          {/* New Section: LTX 2.3 Audio Settings (Conditional) */}
          {userProfile?.videoGeneratorModel === 'ltx2.3' && (
            <div className="p-4 rounded-2xl bg-white/5 border-2 border-dashed border-violet-500/30 space-y-3">
              <span className="text-xs font-black uppercase tracking-wider text-white/90">
                Audio & Voice Clone Settings
              </span>
              
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={audioRefEnabled}
                    onChange={(e) => setAudioRefEnabled(e.target.checked)}
                    className="w-4 h-4 accent-violet-500"
                  />
                  <label className="text-xs text-white/80">Enable Voice Clone</label>
                </div>
                
                <div className={audioRefEnabled ? "opacity-100" : "opacity-40 pointer-events-none"}>
                  <button
                    type="button"
                    onClick={() => audioFileInputRef.current?.click()}
                    disabled={!audioRefEnabled || isGenerating}
                    className="w-full p-3 rounded-xl border border-dashed border-white/20 hover:border-violet-400/50 bg-black/20 text-center transition-all cursor-pointer"
                  >
                    <span className="text-xs text-white/60">
                      {audioRefFile ? 'Audio Terunggah (Klik untuk Ganti)' : 'Upload Audio Reference (~4s)'}
                    </span>
                  </button>
                  <input 
                    type="file" 
                    ref={audioFileInputRef} 
                    className="hidden" 
                    accept="audio/*" 
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) {
                        const reader = new FileReader();
                        reader.onload = (event) => setAudioRefFile(event.target?.result as string);
                        reader.readAsDataURL(file);
                      }
                    }}
                  />

                  <div className="mt-3 space-y-3">
                    <div>
                      <label className="text-[10px] text-white/60">Identity Guidance Scale</label>
                      <input 
                        type="range" min="0" max="10" step="0.1" value={audioRefGuidanceScale}
                        onChange={(e) => setAudioRefGuidanceScale(parseFloat(e.target.value))}
                        className="w-full h-1 bg-white/10 rounded-lg accent-violet-400"
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Section 2: Prompt & AI Prompt Director */}
          <div className="p-4 sm:p-5 rounded-2xl bg-white/5 border border-white/10 space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div>
                <label className="text-xs font-black uppercase tracking-wider text-white/90 flex items-center gap-1.5">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Prompt & Prompt Engineering (Gemini 3.1 Flash Lite)
                </label>
                <p className="text-[11px] text-white/50">
                  Tuliskan ide gerakan / aksi (termasuk aksi intim / eksplisit) dalam Bahasa Indonesia atau Inggris. AI Director akan memformulasikannya menjadi prompt bahasa Inggris sinematik tanpa sensor untuk Image to Video.
                </p>
              </div>

              <button
                type="button"
                onClick={handleEnhancePromptWithAI}
                disabled={isEnhancingPrompt || isGenerating}
                className="self-start sm:self-auto px-3.5 py-1.5 rounded-xl bg-gradient-to-r from-amber-500/20 to-purple-500/20 hover:from-amber-500/30 hover:to-purple-500/30 border border-amber-500/30 text-amber-300 text-xs font-bold flex items-center gap-1.5 transition-all active:scale-95 disabled:opacity-40 shadow-sm cursor-pointer"
              >
                {isEnhancingPrompt ? (
                  <>
                    <div className="w-3.5 h-3.5 border-2 border-amber-400 border-t-transparent rounded-full animate-spin" />
                    <span>Menyusun Prompt (Gemini 3.1)...</span>
                  </>
                ) : (
                  <>
                    <span>✨ AI Director (Gemini 3.1 Flash Lite)</span>
                  </>
                )}
              </button>
            </div>

            {/* User Idea Input */}
            <div>
              <input
                type="text"
                placeholder="Ide gerakan: Contoh 'tersenyum manis' atau aksi intim/eksplisit..."
                value={userIdea}
                onChange={(e) => setUserIdea(e.target.value)}
                disabled={isGenerating}
                className="w-full px-3.5 py-2 text-xs rounded-xl bg-black/40 border border-white/15 text-white placeholder-white/30 outline-none focus:border-amber-400/60 transition-all"
              />
            </div>

            {/* Engineered Video Prompt */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] font-bold text-white/50 uppercase tracking-wider">
                  Hasil Prompt Sinematik :
                </span>
                <span className="text-[10px] font-mono text-white/40">
                  {prompt.length} karakter
                </span>
              </div>
              <textarea
                rows={3}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                disabled={isGenerating}
                placeholder="English cinematic video prompt..."
                className="w-full p-3 text-xs font-mono rounded-xl bg-black/50 border border-white/15 text-emerald-300 placeholder-white/30 outline-none focus:border-emerald-400/60 transition-all resize-y custom-scrollbar leading-relaxed"
              />
            </div>
          </div>

          {/* Section: Slider Pengatur Durasi Video */}
          <div className="p-4 sm:p-5 rounded-2xl bg-white/5 border border-white/10 space-y-3.5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-[0_0_10px_rgba(34,211,238,0.5)]"></span>
                <label className="text-xs font-black uppercase tracking-wider text-white/90 flex items-center gap-1.5">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Durasi Video (Detik)
                </label>
              </div>
              <div className="flex items-center gap-2">
                <span className="px-3 py-1 rounded-xl bg-cyan-500/20 border border-cyan-500/30 text-cyan-300 font-mono font-black text-sm shadow-sm">
                  {duration.toFixed(1)}s
                </span>
                <span className="text-[10px] text-white/40 font-mono hidden sm:inline">
                  (~{Math.round(duration * 64)} frames @ 64 FPS)
                </span>
              </div>
            </div>

            {/* Range Slider Track */}
            <div className="space-y-2">
              <div className="relative flex items-center">
                <input
                  type="range"
                  min={1.0}
                  max={10.0}
                  step={0.5}
                  value={duration}
                  onChange={(e) => setDuration(parseFloat(e.target.value))}
                  disabled={isGenerating}
                  className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer accent-cyan-400 focus:outline-none transition-all disabled:opacity-40"
                  style={{
                    accentColor: '#22d3ee'
                  }}
                />
              </div>

              {/* Slider Min-Max indicators & Quick Preset Buttons */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pt-1">
                <div className="flex items-center gap-1.5 text-[10px] text-white/40 font-mono">
                  <span>1.0s</span>
                  <span>•</span>
                  <span>Default 5.0s</span>
                  <span>•</span>
                  <span>Max 10.0s</span>
                </div>

                <div className="flex items-center gap-1.5 flex-wrap">
                  <span className="text-[10px] font-bold text-white/40 uppercase tracking-wider mr-1">
                    Preset:
                  </span>
                  {[2.0, 3.5, 5.0, 7.5, 10.0].map((presetSec) => (
                    <button
                      key={presetSec}
                      type="button"
                      onClick={() => setDuration(presetSec)}
                      disabled={isGenerating}
                      className={`px-2.5 py-1 rounded-lg text-[10px] font-mono font-bold transition-all cursor-pointer ${
                        duration === presetSec
                          ? 'bg-cyan-400 text-black shadow-md font-black scale-105'
                          : 'bg-white/10 text-white/70 hover:bg-white/20 hover:text-white'
                      } disabled:opacity-40`}
                    >
                      {presetSec.toFixed(1)}s
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <p className="text-[10px] text-white/40 leading-relaxed italic">
              💡 Catatan: Durasi lebih panjang menghasilkan klip aksi yang lebih lengkap namun membutuhkan waktu antrean komputasi ZeroGPU sedikit lebih lama.
            </p>
          </div>

          {/* Section 3: Technical Specs & Status Presets */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div className="p-3 rounded-xl bg-white/5 border border-white/10 flex flex-col gap-0.5">
              <span className="text-[10px] font-bold uppercase tracking-wider text-white/40">Durasi Video</span>
              <span className="font-mono font-black text-cyan-300 text-sm">{duration.toFixed(1)} Detik</span>
              <span className="text-[9px] text-cyan-400/80">Slider Aktif</span>
            </div>
            <div className="p-3 rounded-xl bg-white/5 border border-white/10 flex flex-col gap-0.5">
              <span className="text-[10px] font-bold uppercase tracking-wider text-white/40">Smoothness</span>
              <span className="font-mono font-black text-white text-sm">64 FPS</span>
              <span className="text-[9px] text-emerald-400">Fixed Preset</span>
            </div>
            <div className="p-3 rounded-xl bg-white/5 border border-white/10 flex flex-col gap-0.5">
              <span className="text-[10px] font-bold uppercase tracking-wider text-white/40">Backend Engine</span>
              <span className="font-mono font-black text-violet-300 truncate text-xs">Image-to-Video</span>
              <span className="text-[9px] text-white/50">ZeroGPU</span>
            </div>
            <div className="p-3 rounded-xl bg-white/5 border border-white/10 flex flex-col gap-0.5">
              <span className="text-[10px] font-bold uppercase tracking-wider text-white/40">Token Status</span>
              <span className="font-mono font-bold text-white truncate text-xs">
                {userProfile?.hfTokens ? 'Token Pengguna Terpasang' : 'Publik / Default'}
              </span>
              <span className="text-[9px] text-white/50">Auto-Rotate Quota</span>
            </div>
          </div>

          {/* Section 4: Live Generation Progress */}
          {isGenerating && (
            <div className="p-4 rounded-2xl bg-gradient-to-r from-violet-950/40 via-purple-950/40 to-black/60 border border-violet-500/30 space-y-3 animate-in fade-in duration-300">
              <div className="flex items-center justify-between text-xs font-bold text-violet-300">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 border-2 border-violet-400 border-t-transparent rounded-full animate-spin" />
                  <span>{generationStatus || "Memproses tugas video..."}</span>
                </div>
                <span className="font-mono text-white/60">{elapsedSeconds}s</span>
              </div>
              <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
                <div 
                  className="h-full rounded-full transition-all duration-300 animate-pulse"
                  style={{
                    width: `${Math.min(95, elapsedSeconds * 2.5)}%`,
                    backgroundColor: themeHex
                  }}
                />
              </div>
              <p className="text-[10px] text-white/50 leading-relaxed italic">
                Tips: Merender video dengan resolusi tinggi pada 64 FPS. Durasi proses rata-rata 30-90 detik tergantung antrean ZeroGPU.
              </p>
            </div>
          )}

          {/* Section 5: Rendered Video Output */}
          {generatedVideoUrl && (
            <div className="p-5 rounded-2xl bg-black/70 border border-emerald-500/40 space-y-4 animate-in zoom-in-95 duration-300">
              <div className="flex items-center justify-between">
                <span className="text-xs font-black uppercase tracking-wider text-emerald-400 flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping" />
                  Video Berhasil Dibuat! (MP4 · 64 FPS · {duration.toFixed(1)}s)
                </span>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={handleDownload}
                    className="px-3 py-1 rounded-xl bg-white/10 hover:bg-white/20 text-white text-xs font-bold flex items-center gap-1.5 transition-all cursor-pointer"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a2 2 0 002 2h12a2 2 0 002-2v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                    Unduh MP4
                  </button>
                  <button
                    type="button"
                    onClick={handleInsertToChat}
                    disabled={isSavingToChat}
                    className="px-3.5 py-1.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-60 text-white text-xs font-black uppercase tracking-wider flex items-center gap-1.5 transition-all shadow-lg cursor-pointer"
                  >
                    {isSavingToChat ? (
                      <>
                        <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                        <span>Menganalisa...</span>
                      </>
                    ) : (
                      <>
                        <span>💬</span>
                        <span>Simpan ke Chat</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Video Player */}
              <div className="relative rounded-2xl overflow-hidden bg-black aspect-video max-h-96 mx-auto flex items-center justify-center border border-white/10 shadow-2xl">
                <video
                  ref={videoPlayerRef}
                  src={generatedVideoUrl}
                  controls
                  autoPlay
                  loop
                  playsInline
                  className="w-full h-full object-contain"
                />
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer / Action CTA */}
        <div className="sticky bottom-0 z-20 px-6 py-4 bg-zinc-950/90 backdrop-blur-xl border-t border-white/10 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="text-[11px] text-white/50 text-center sm:text-left">
            <span>Image to Video · </span>
            <span className="text-white/70">Start Frame (wajib)</span>
            <span> + </span>
            <span className="text-white/40">End Frame (opsional)</span>
          </div>

          <div className="flex items-center gap-2 w-full sm:w-auto">
            <button
              type="button"
              onClick={onClose}
              disabled={isGenerating}
              className="flex-1 sm:flex-none px-4 py-2.5 rounded-2xl bg-white/5 hover:bg-white/10 text-white text-xs font-bold transition-all disabled:opacity-40 cursor-pointer"
            >
              Tutup
            </button>

            <button
              type="button"
              onClick={handleGenerateVideo}
              disabled={isGenerating || !startImage}
              className="flex-1 sm:flex-none px-6 py-2.5 rounded-2xl text-white text-xs font-black uppercase tracking-wider shadow-xl transition-all active:scale-95 disabled:opacity-40 cursor-pointer flex items-center justify-center gap-2"
              style={{
                background: `linear-gradient(135deg, ${themeHex}, #8b5cf6)`,
                boxShadow: `0 4px 20px ${themeHex}50`
              }}
            >
              {isGenerating ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  <span>Sedang Merender ({elapsedSeconds}s)...</span>
                </>
              ) : (
                <>
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>Mulai Render Video</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WanVideoModal;
