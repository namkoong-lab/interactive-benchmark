import { Link, useLocation } from "react-router-dom";

// Icon components
const HomeIcon = () => (
  <svg
    className="w-5 h-5"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"
    />
  </svg>
);

const PaperIcon = () => (
  <svg
    className="w-5 h-5"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
    />
  </svg>
);

const CodeIcon = () => (
  <svg
    className="w-5 h-5"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"
    />
  </svg>
);

const SubmitIcon = () => (
  <svg
    className="w-5 h-5"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
    />
  </svg>
);

const LeaderboardIcon = () => (
  <svg
    className="w-5 h-5"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
    />
  </svg>
);

export default function Hero() {
  const location = useLocation();

  const navItems = [
    { name: "Home", href: "/", icon: HomeIcon, external: false },
    {
      name: "Paper",
      href: "https://arxiv.org/abs/XXXX.XXXXX",
      icon: PaperIcon,
      external: true,
    },
    {
      name: "Code",
      href: "https://github.com/namkoong-lab/data-recipes",
      icon: CodeIcon,
      external: true,
    },
    { name: "Submit", href: "/submit", icon: SubmitIcon, external: false },
    {
      name: "Leaderboard",
      href: "/leaderboard",
      icon: LeaderboardIcon,
      external: false,
    },
  ];

  const isActive = (path: string) => location.pathname === path;

  return (
    <div className="bg-gradient-to-r from-indigo-600 via-blue-600 to-cyan-600 text-white">
      {/* Hero Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 md:py-28">
        <div className="text-center">
          {/* Main Title */}
          <Link
            to="/"
            className="inline-block hover:opacity-90 transition-opacity"
          >
            <h1 className="text-6xl md:text-8xl font-extrabold mb-8 tracking-tight drop-shadow-lg">
              Agent Thesis
            </h1>
          </Link>

          {/* Subtitle/Description */}
          <p className="text-2xl md:text-3xl mb-10 text-white/95 max-w-4xl mx-auto font-light">
            A Groundbreaking Benchmark for Evaluating AI Agents
          </p>

          {/* Authors */}
          <div className="mb-8">
            <p className="text-lg md:text-xl text-white/90 font-medium">
              Author Name<sup className="text-sm">1</sup>, Collaborator Name
              <sup className="text-sm">2</sup>, Another Author
              <sup className="text-sm">1,3</sup>
            </p>
            <p className="text-base md:text-lg text-white/75 mt-3">
              <sup className="text-xs">1</sup>Institution Name ·{" "}
              <sup className="text-xs">2</sup>University Name ·{" "}
              <sup className="text-xs">3</sup>Research Lab
            </p>
          </div>
        </div>
      </div>

      {/* Navigation Buttons */}
      <div className="pb-8 pt-4">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-wrap justify-center items-center gap-3">
            {navItems.map((item) => {
              const Icon = item.icon;
              const buttonClasses = `px-8 py-4 rounded-lg font-semibold transition-all duration-200 shadow-md hover:shadow-xl transform hover:-translate-y-0.5 flex items-center gap-2 ${
                isActive(item.href)
                  ? "bg-white text-indigo-700"
                  : "bg-white/20 hover:bg-white/30 backdrop-blur-md border border-white/40 hover:border-white/60"
              }`;

              return item.external ? (
                <a
                  key={item.name}
                  href={item.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={buttonClasses}
                >
                  <Icon />
                  <span>{item.name}</span>
                </a>
              ) : (
                <Link key={item.name} to={item.href} className={buttonClasses}>
                  <Icon />
                  <span>{item.name}</span>
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
