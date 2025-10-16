# Agent Thesis

A modern, responsive website for presenting AI agent research, built with React, TypeScript, and Tailwind CSS.

## 🚀 Features

- **Hero Section**: Eye-catching landing page with gradient design
- **About**: Project overview and research description
- **Service Demo**: Video demonstrations of services
- **Agent Demo**: Video demonstrations of agent capabilities
- **Citation**: BibTeX citation for research papers
- **Quick Start**: Getting started guides
- **Contact**: Contact information and links
- **Submit**: Instructions for submitting model results
- **Leaderboard**: Compare performance metrics across different models

## 🛠️ Tech Stack

- **Framework**: React 18
- **Language**: TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **Routing**: React Router DOM

## 📦 Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## 🌐 Development

The development server will start at `http://localhost:5173/`

## 📁 Project Structure

```
agent-thesis/
├── src/
│   ├── components/       # Reusable components
│   │   ├── home/        # Home page components
│   │   ├── submit/      # Submit page components
│   │   └── leaderboard/ # Leaderboard components
│   ├── pages/           # Page components
│   ├── App.tsx          # Main app component
│   └── main.tsx         # Entry point
├── results/             # Model results directory
│   └── example-model/   # Example model template
├── public/              # Static assets
└── index.html           # HTML template
```

## 📊 Adding Model Results

To add your model to the leaderboard:

1. Create a folder in `results/` with your model name (use lowercase with hyphens)
2. Add a `results.json` file following the format in `results/example-model/`
3. Submit a pull request

See `results/README.md` for detailed instructions.

## 🎨 Customization

- **Colors**: Edit `tailwind.config.ts` to change the color scheme
- **Content**: Update text and images in component files
- **Styling**: Modify Tailwind classes or add custom CSS in `index.css`

## 📄 License

[Your License Here]

## 📧 Contact

For questions or contributions, please contact: contact@agent-thesis.com
