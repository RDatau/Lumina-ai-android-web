import React, { useState } from 'react';

interface VideoInfoModalProps {
  isOpen: boolean;
  onClose: () => void;
  videoUrl: string | null;
  videoPrompt?: string;
  caption?: string;
  duration?: number;
  timestamp?: number;
  isBackgroundDark?: boolean;
  themeHex?: string;
  agentName?: string;
}

export const VideoInfoModal: React.FC<VideoInfoModalProps> = ({
  isOpen,
  onClose,
  videoUrl,
  videoPrompt,
  caption,
  duration = 5.0,
  timestamp,
  isBackgroundDark = true,
  themeHex = '#8b5cf6',
  agentName
}) => {
  const [copiedPrompt, setCopiedPrompt] = useState(false);

  if (!isOpen || !videoUrl) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(videoPrompt || caption || '');
    setCopiedPrompt(true);
    setTimeout(() => setCopiedPrompt(false), 2000);
  };

  const formattedDate = timestamp ? new Date(timestamp).toLocaleString() : new Date().toLocaleString();

  return (
    <div className={`fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fade-in ${typeof window !== 'undefined' && !!(window as any).electron?.isElectron ? 'pt-8' : ''}`} onClick={onClose}>
      <div 
        className={`relative w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-3xl p-6 shadow-2xl border ${isBackgroundDark ? 'bg-zinc-900/95 border-white/20 text-white' : 'bg-white/95 border-black/15 text-zinc-900'} backdrop-blur-xl`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between pb-4 mb-4 border-b border-white/10">
          <div className="flex items-center gap-2">
            <span className="px-3 py-1 rounded-full bg-violet-600/30 text-violet-300 border border-violet-500/50 text-xs font-black uppercase tracking-wider">
              Video Info
            </span>
          </div>
          <button 
            type="button"
            onClick={onClose}
            className={`p-2 rounded-full ${isBackgroundDark ? 'hover:bg-white/10 text-white/70' : 'hover:bg-black/10 text-black/70'} transition-all`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="space-y-6">
          {/* Video Preview */}
          <div className="relative overflow-hidden rounded-2xl bg-black border border-white/20 shadow-lg">
            <video 
              src={videoUrl} 
              autoPlay
              muted
              loop
              playsInline
              className="w-full max-h-72 object-contain rounded-2xl bg-black" 
            />
          </div>

          {/* Details Grid */}
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className={`p-3 rounded-xl ${isBackgroundDark ? 'bg-white/5 border border-white/10' : 'bg-black/5 border border-black/10'}`}>
              <span className="opacity-60 block font-bold mb-1 uppercase tracking-wider text-[10px]">Durasi & FPS</span>
              <span className="font-mono font-bold text-violet-400">{duration.toFixed(1)}s · 64 FPS</span>
            </div>
            <div className={`p-3 rounded-xl ${isBackgroundDark ? 'bg-white/5 border border-white/10' : 'bg-black/5 border border-black/10'}`}>
              <span className="opacity-60 block font-bold mb-1 uppercase tracking-wider text-[10px]">Waktu Dibuat</span>
              <span className="font-mono font-bold">{formattedDate}</span>
            </div>
            {agentName && (
              <div className={`p-3 rounded-xl col-span-2 ${isBackgroundDark ? 'bg-white/5 border border-white/10' : 'bg-black/5 border border-black/10'}`}>
                <span className="opacity-60 block font-bold mb-1 uppercase tracking-wider text-[10px]">Karakter / Agen</span>
                <span className="font-bold text-emerald-400">{agentName}</span>
              </div>
            )}
          </div>

          {/* Prompt / Commentary Section */}
          {(videoPrompt || caption) && (
            <div className={`p-4 rounded-2xl ${isBackgroundDark ? 'bg-white/5 border border-white/10' : 'bg-black/5 border border-black/10'}`}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-black uppercase tracking-wider opacity-70">Prompt / Komentar Video</span>
                <button
                  type="button"
                  onClick={handleCopy}
                  className="px-3 py-1 rounded-lg bg-violet-600/30 hover:bg-violet-600/50 text-violet-200 text-xs font-bold transition-all flex items-center gap-1.5"
                >
                  {copiedPrompt ? (
                    <>
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5 text-emerald-400" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" /></svg>
                      <span className="text-emerald-300">Tersalin!</span>
                    </>
                  ) : (
                    <>
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7v8a2 2 0 002 2h6M8 7V5a2 2 0 012-2h4.586a1 1 0 01.707.293l4.414 4.414a1 1 0 01.293.707V15a2 2 0 01-2 2h-2M8 7H6a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2v-2" /></svg>
                      <span>Salin Prompt</span>
                    </>
                  )}
                </button>
              </div>
              <p className="text-sm font-mono whitespace-pre-wrap leading-relaxed opacity-90 select-text">
                {videoPrompt || caption}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default VideoInfoModal;
