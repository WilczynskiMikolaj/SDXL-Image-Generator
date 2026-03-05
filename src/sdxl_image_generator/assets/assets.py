FILE_EXPLORER_SVG = """
<svg viewBox="0 0 128 128" xmlns="http://www.w3.org/2000/svg">
  <rect x="16" y="36" width="80" height="60" rx="12" fill="url(#folderGradient)"/>
  <path d="M16 44C16 38 20 34 26 34H50C54 34 56 36 58 40L62 44H16Z" fill="url(#tabGradient)"/>
  <circle cx="78" cy="78" r="20" fill="url(#lensGradient)" stroke="#0F172A" stroke-width="4"/>
  <rect x="90" y="90" width="26" height="8" rx="4"
        transform="rotate(45 90 90)" fill="#1E293B"/>

  <defs>
    <linearGradient id="folderGradient" x1="16" y1="36" x2="96" y2="96">
      <stop stop-color="#2563EB"/>
      <stop offset="1" stop-color="#22C55E"/>
    </linearGradient>

    <linearGradient id="tabGradient" x1="16" y1="34" x2="62" y2="44">
      <stop stop-color="#F59E0B"/>
      <stop offset="1" stop-color="#F97316"/>
    </linearGradient>

    <radialGradient id="lensGradient"
        gradientTransform="translate(72 72) rotate(45) scale(28)">
      <stop stop-color="#38BDF8"/>
      <stop offset="1" stop-color="#6366F1"/>
    </radialGradient>
  </defs>
</svg>
"""