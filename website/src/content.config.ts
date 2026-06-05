import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const sections = defineCollection({
  loader: glob({
    pattern: '**/*.{md,mdx}',
    base: './src/content/sections',
  }),
  schema: z.object({
    title: z.string(),
    order: z.number().optional(),
    description: z.string().optional(),
  }),
});

export const collections = {
  sections,
};
