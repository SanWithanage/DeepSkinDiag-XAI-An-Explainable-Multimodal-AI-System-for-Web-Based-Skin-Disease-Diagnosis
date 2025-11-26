/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "media",
  theme: {
    extend: {
      fontFamily: { sans: ["Inter", "ui-sans-serif", "system-ui"] },
      colors: {
        brand: {
          50: "#eef2ff",
          100: "#e0e7ff",
          500: "#6366f1",
          600: "#4f46e5",
          700: "#4338ca",
        },
      },
      boxShadow: {
        soft: "0 10px 30px -12px rgba(0,0,0,0.15)",
        ring: "0 0 0 1px rgba(255,255,255,0.08) inset, 0 1px 0 0 rgba(255,255,255,0.06) inset",
      },
      borderRadius: { xl2: "1.25rem" },
      keyframes: {
        fadeUp: { "0%": { opacity: 0, transform: "translateY(6px)" }, "100%": { opacity: 1, transform: "translateY(0)" } },
      },
      animation: { fadeUp: "fadeUp .4s ease-out both" },
    },
  },
  plugins: [],
};
