export function Waveform() {
  const BAR_COUNT = 60;
  const bars = Array.from({ length: BAR_COUNT });
  const mid = BAR_COUNT / 2;

  return (
    <div className="vr-waveform-overlay">
      <div className="vr-waveform-bars">
        {bars.map((_, i) => {
          const distFromMid = Math.abs(i - mid) / mid;
          const baseHeight = 100 * (1 - Math.pow(distFromMid, 1.6));
          return (
            <span
              key={i}
              className="vr-wave-bar"
              style={{
                height: `${Math.max(8, baseHeight)}%`,
                animationDelay: `${(i * 0.04) % 1.2}s`,
                animationDuration: `${0.7 + (i % 4) * 0.15}s`,
              }}
            />
          );
        })}
      </div>
    </div>
  );
}