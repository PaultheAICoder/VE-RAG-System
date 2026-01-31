interface BridgeLogoProps {
  size?: number;
  className?: string;
}

export function BridgeLogo({ size = 120, className = '' }: BridgeLogoProps) {
  return (
    <svg
      width={size}
      height={size * 0.55}
      viewBox="0 0 120 66"
      fill="none"
      className={className}
    >
      <path d="M8 32 Q60 2 112 32" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round"/>
      <path d="M20 58 L20 24 L26 16 L32 24 L32 58" stroke="currentColor" strokeWidth="1.5" fill="none"/>
      <line x1="26" y1="16" x2="26" y2="58" stroke="currentColor" strokeWidth="1.5"/>
      <line x1="20" y1="28" x2="32" y2="28" stroke="currentColor" strokeWidth="1.5"/>
      <line x1="20" y1="36" x2="32" y2="36" stroke="currentColor" strokeWidth="1.5"/>
      <line x1="20" y1="44" x2="32" y2="44" stroke="currentColor" strokeWidth="1.5"/>
      <line x1="20" y1="52" x2="32" y2="52" stroke="currentColor" strokeWidth="1.5"/>
      <rect x="22" y="10" width="8" height="6" stroke="currentColor" strokeWidth="1.5" fill="none"/>
      <line x1="26" y1="6" x2="26" y2="10" stroke="currentColor" strokeWidth="1"/>
      <path d="M88 58 L88 24 L94 16 L100 24 L100 58" stroke="currentColor" strokeWidth="1.5" fill="none"/>
      <line x1="94" y1="16" x2="94" y2="58" stroke="currentColor" strokeWidth="1.5"/>
      <line x1="88" y1="28" x2="100" y2="28" stroke="currentColor" strokeWidth="1.5"/>
      <line x1="88" y1="36" x2="100" y2="36" stroke="currentColor" strokeWidth="1.5"/>
      <line x1="88" y1="44" x2="100" y2="44" stroke="currentColor" strokeWidth="1.5"/>
      <line x1="88" y1="52" x2="100" y2="52" stroke="currentColor" strokeWidth="1.5"/>
      <rect x="90" y="10" width="8" height="6" stroke="currentColor" strokeWidth="1.5" fill="none"/>
      <line x1="94" y1="6" x2="94" y2="10" stroke="currentColor" strokeWidth="1"/>
      <line x1="40" y1="12" x2="40" y2="46" stroke="currentColor" strokeWidth="1.2"/>
      <line x1="50" y1="7" x2="50" y2="46" stroke="currentColor" strokeWidth="1.2"/>
      <line x1="60" y1="4" x2="60" y2="46" stroke="currentColor" strokeWidth="1.2"/>
      <line x1="70" y1="7" x2="70" y2="46" stroke="currentColor" strokeWidth="1.2"/>
      <line x1="80" y1="12" x2="80" y2="46" stroke="currentColor" strokeWidth="1.2"/>
      <rect x="36" y="46" width="48" height="6" stroke="currentColor" strokeWidth="1.5" fill="none" rx="1"/>
      <line x1="4" y1="58" x2="116" y2="58" stroke="currentColor" strokeWidth="2"/>
    </svg>
  );
}
