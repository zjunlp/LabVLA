/** Tailwind v3 build config — mirrors the former cdn.tailwindcss.com inline config.
 *  Rebuild after editing markup:
 *    npx -y tailwindcss@3 -c tailwind.config.js -i tailwind.input.css -o assets/css/tailwind.css --minify
 */
module.exports = {
  content: ['./index.html', './assets/js/main.js'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Source Sans Pro"', 'Lato', 'Verdana', 'Helvetica', 'sans-serif'],
        display: ['"Source Sans Pro"', 'Lato', 'Verdana', 'Helvetica', 'sans-serif'],
        mono: ['"Source Code Pro"', 'monospace'],
      },
      colors: {
        ink: { 950: '#06070A', 900: '#0B0D12', 800: '#11141B', 700: '#1A1E27' },
        accent: { DEFAULT: '#7C5CFF', soft: '#A78BFA', cyan: '#22D3EE', lime: '#A3E635' },
      },
      boxShadow: {
        glow: '0 0 60px -10px rgba(124,92,255,0.45)',
      },
    },
  },
};
