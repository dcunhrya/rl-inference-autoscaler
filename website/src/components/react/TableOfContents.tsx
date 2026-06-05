import { useEffect, useState } from 'react';

export interface TableOfContentsProps {
  sections: { id: string; title: string }[];
}

export default function TableOfContents({ sections }: TableOfContentsProps) {
  const [activeId, setActiveId] = useState<string | null>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            const id = entry.target.getAttribute('id');
            if (id) setActiveId(id);
            break;
          }
        }
      },
      {
        rootMargin: '-80px 0px -80% 0px',
        threshold: 0,
      },
    );

    const elements = sections
      .map(({ id }) => document.getElementById(id))
      .filter(Boolean) as HTMLElement[];
    elements.forEach((el) => observer.observe(el));

    return () => observer.disconnect();
  }, [sections]);

  return (
    <nav aria-label="Table of contents" className="space-y-1">
      <h2 className="text-xs font-semibold uppercase tracking-wider text-ink-light mb-3">
        On this page
      </h2>
      <ul className="space-y-1">
        {sections.map(({ id, title }) => (
          <li key={id}>
            <a
              href={`#${id}`}
              className={`block text-sm py-1.5 px-2 rounded-md transition-colors ${
                activeId === id
                  ? 'bg-accent/10 text-accent font-medium border-l-2 border-accent'
                  : 'text-ink-muted hover:text-accent hover:bg-cream-muted/80'
              }`}
            >
              {title}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}
