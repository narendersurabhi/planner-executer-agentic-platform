import { Fraunces, Manrope } from "next/font/google";
import "./globals.css";
import { AppThemeProvider } from "./lib/theme";

const displayFont = Fraunces({
  subsets: ["latin"],
  variable: "--font-display",
  display: "swap"
});

const bodyFont = Manrope({
  subsets: ["latin"],
  variable: "--font-body",
  display: "swap"
});

export const metadata = {
  title: "AI Workflow Workspace",
  description:
    "A workflow platform for designing, running, and monitoring AI-powered business automations."
};

export default function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${displayFont.variable} ${bodyFont.variable} min-h-screen font-body`}>
        <AppThemeProvider>
          <div className="w-full px-6 py-8">{children}</div>
        </AppThemeProvider>
      </body>
    </html>
  );
}
