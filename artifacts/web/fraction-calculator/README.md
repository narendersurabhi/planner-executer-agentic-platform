# Friendly Fraction Calculator (Beta)

A tiny, self-contained web app for doing fraction addition, multiplication, and division. It shows kid-friendly steps and suggests similar practice problems with answers.

## Features
- Add, multiply, or divide two fractions
- Clear, step-by-step explanation (written for a 10-year-old)
- Answers shown as a simplified fraction and as a mixed number
- 3 similar practice problems with answers
- Works offline, no tracking, no external libraries

## How to use
1. Open `index.html` in your web browser (double-click it).
2. Enter Fraction A and Fraction B (use positive whole numbers).
3. Choose an operation (+, ×, ÷).
4. Click "Calculate".
5. Read the steps and try the similar problems.

## Files
- `index.html` — page layout and UI
- `styles.css` — simple, accessible styling
- `app.js` — fraction math, explanations, and similar-problem generation

## Beta program
- Version: v0.1 (Beta)
- Scope: core fraction operations (addition, multiplication, division) and explanations
- Feedback: please report clarity/usability issues and any incorrect steps or edge cases

## Security review
- No external scripts, no network calls, no data collection
- Inputs limited to positive integers; denominators cannot be zero
- Client-side only; suitable for offline classroom or home use

## Docs
- The code is commented and organized into utilities (gcd/lcm/simplify), operations with steps, and DOM wiring.
- For new operations (like subtraction), follow the same pattern: compute, explain steps in friendly language, then render.

## Constraints acknowledged
- No external agencies or libraries used.

## License
- Provided as-is for educational use.
