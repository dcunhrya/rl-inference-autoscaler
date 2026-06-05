export interface BlogAuthor {
  name: string;
  affiliation: string;
}

export interface BlogHeaderProps {
  title?: string;
  authors: BlogAuthor[];
  publishedDate: string;
}

export default function BlogHeader({
  title,
  authors,
  publishedDate,
}: BlogHeaderProps) {
  return (
    <header className="mb-10 pb-8 border-b border-border not-prose">
      {title ? (
        <h1 className="text-3xl sm:text-4xl font-bold text-accent leading-tight mb-8">
          {title}
        </h1>
      ) : null}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 text-sm">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-ink-light mb-2">
            Authors
          </p>
          <ul className="space-y-1">
            {authors.map((author) => (
              <li key={author.name} className="text-ink font-medium">
                {author.name}
              </li>
            ))}
          </ul>
        </div>
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-ink-light mb-2">
            Affiliations
          </p>
          <ul className="space-y-1">
            {authors.map((author) => (
              <li key={`${author.name}-aff`} className="text-ink-muted">
                {author.affiliation || '—'}
              </li>
            ))}
          </ul>
        </div>
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-ink-light mb-2">
            Published
          </p>
          <p className="text-ink">{publishedDate}</p>
        </div>
      </div>
    </header>
  );
}
